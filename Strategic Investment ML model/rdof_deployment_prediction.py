import requests
import pandas as pd
import time
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

class CensusDataProcessor:
    def __init__(self, api_key, input_file):
        self.api_key = api_key
        self.input_file = input_file
        
        # Census variables with readable names
        self.variable_names = {
            'B01003_001E': 'total_population',
            'B01001_002E': 'male_population',
            'B01001_026E': 'female_population',
            'B02001_002E': 'white_alone',
            'B02001_003E': 'black_alone',
            'B02001_004E': 'american_indian_alaska_native',
            'B02001_005E': 'asian_alone',
            'B02001_006E': 'pacific_islander',
            'B02001_007E': 'other_race',
            'B02001_008E': 'multi_race',
            'B03003_003E': 'hispanic_latino',
            'B01001_040E': 'population_under_18',
            'B21001_001E': 'population_18_plus',
            'B01001_043E': 'population_65_plus',
            'B09010_001E': 'population_in_households',
            'B11010_001E': 'total_households',
            'B11011_001E': 'total_families',
            'B15003_001E': 'population_25_plus',
            'B20004_002E': 'male_high_school',
            'B20004_003E': 'male_bachelors',
            'B20004_004E': 'male_masters',
            'B20004_005E': 'male_professional',
            'B20004_006E': 'male_doctorate',
            'B20004_007E': 'female_high_school',
            'B20004_008E': 'female_bachelors',
            'B20004_009E': 'female_masters',
            'B20004_010E': 'female_professional',
            'B20004_011E': 'female_doctorate',
            'B25001_001E': 'housing_units',
            'B25002_001E': 'owner_occupied',
            'B25002_002E': 'renter_occupied',
            'B25004_001E': 'vacant_units',
            'B19013_001E': 'median_household_income',
            'B19113_001E': 'median_family_income',
            'B25077_001E': 'median_home_value',
            'B25064_001E': 'median_gross_rent'
        }
        
        # Get API variables from the keys
        self.variables = list(self.variable_names.keys())

    def get_acs_data(self, state, county, tract, block_group):
        """Retrieves ACS data for specified block group and variables."""
        endpoint = "https://api.census.gov/data/2022/acs/acs5"
        params = {
            "get": ",".join(self.variables),
            "for": "block group:" + block_group,
            "in": f"state:{state} county:{county} tract:{tract}",
            "key": self.api_key
        }
        headers = {
            "User-Agent": "Census Data Retrieval Application"
        }

        try:
            response = requests.get(endpoint, params=params, headers=headers)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

    def process_data(self, sample_size=1000):
        """Process census data for random sample of unique census blocks."""
        try:
            # Read and verify file contents
            df = pd.read_csv(self.input_file, 
                        dtype={'census_id': str,
                               'auction_id': str,
                               'bidder': str,
                               'frn': str,
                               'block_id': str,
                               'item': str,
                               'state': str,
                               'county': str})
        
            print(f"\nTotal records in file: {len(df)}")
            print("\nFirst few records:")
            print(df[['bidder', 'census_id']].head())
            
            # Take random sample
            df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            # Format census_ids to ensure 12 digits
            df_sample['census_id'] = df_sample['census_id'].apply(
                lambda x: str(int(float(x))).zfill(12) if x and x.strip() else None
            )
            
            print("\nSample of processed census_ids:")
            print(df_sample[['bidder', 'census_id']].head())
            print("\nProcessing random sample of", sample_size, "census blocks")
            
            # Dictionary to store census API results
            results = defaultdict(dict)
            
            # Process sampled records
            for _, row in tqdm(df_sample.iterrows()):
                census_id = row['census_id']
                try:
                    # Parse components
                    state = census_id[:2]
                    county = census_id[2:5]
                    tract = census_id[5:11]
                    block_group = census_id[11]
                    
                    data = self.get_acs_data(state, county, tract, block_group)
                    
                    if data and len(data) > 1:
                        header = data[0]
                        values = data[1]
                        
                        # Store results
                        result_dict = {}
                        for h, v in zip(header, values):
                            if h in self.variables:
                                if v == 'None' or v is None:
                                    result_dict[h] = None
                                else:
                                    try:
                                        result_dict[h] = float(v)
                                    except:
                                        result_dict[h] = v
                        
                        results[census_id] = result_dict
                
                except Exception as e:
                    print(f"Error processing census_id {census_id}: {str(e)}")
                
                time.sleep(0.05)
            
            # Convert census results to DataFrame
            results_df = pd.DataFrame.from_dict(results, orient='index')
            
            # Rename census columns to readable names
            results_df = results_df.rename(columns=self.variable_names)
            
            print("\nShape of census results:", results_df.shape)
            print("Census columns collected:", results_df.columns.tolist())
            
            if not results_df.empty:
                # Merge census data with original sample records
                final_df = df_sample.merge(results_df, left_on='census_id', right_index=True, how='left')
                
                # Save final results
                output_file = f"census_data_results_sample_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                final_df.to_csv(output_file, index=False)
                print(f"\nSaved final results to: {output_file}")
                
                return final_df
            else:
                raise Exception("No data was collected from the Census API")
                
        except Exception as e:
            print(f"Error in process_data: {str(e)}")
            raise

def validate_input_file(file_path):
    """Validate that the input file exists and has the required columns."""
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        df_headers = pd.read_csv(file_path, nrows=0)
        required_columns = ['bidder', 'census_id']
        missing_columns = [col for col in required_columns if col not in df_headers.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return True
    except Exception as e:
        print(f"Input file validation error: {str(e)}")
        raise

def main():
    api_key = "75120b43159fbf4df85fde7f28114bfb8d39dea7"
    input_file = "Assigned_census_blocks_by_winner_text.csv"
    
    try:
        if validate_input_file(input_file):
            print(f"Input file validated: {input_file}")
            
        processor = CensusDataProcessor(api_key, input_file)
        result_df = processor.process_data(sample_size=1000)
        print(f"Processing complete. Shape of final dataset: {result_df.shape}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()