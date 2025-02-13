"""This script joins the category information from the wiki_id_categories dataset to the wiki_all_label dataset.
The wiki_all_label dataset contains the text and labels for each Wikipedia article, while the wiki_id_categories dataset    
contains the categories for each Wikipedia article. This script adds the categories to the wiki_all_label dataset.
More info: https://microsoftapc-my.sharepoint.com/:w:/g/personal/t-asutradhar_microsoft_com/Ec7UTAvWVhpKvBiDSF8fLIcBLtwTdKzqb7_L-lD2HSNsGw?e=TeLmgB"""



import pandas as pd
from datasets import load_from_disk, Dataset
import time


# Load the datasets
wiki_all_label = load_from_disk('wiki_all_label_dataset.arrow') # Replace with the path to your dataset with labels
print("Size of wiki_all_label: ", len(wiki_all_label))
wiki_id_categories = load_from_disk('wiki_id_categories_from_dump.arrow') # Replace with the path to your dataset with categories
print("Size of wiki_id_categories: ", len(wiki_id_categories))
print("Datasets loaded successfully.")
current_date_time = time.strftime('%Y%m%d%H%M%S')
output_file = f'joined_dataset_{current_date_time}.arrow'

# wiki_all_label = wiki_all_label.select(range(1000))  # Limit the number of rows for testing

# Create a dictionary from the categories dataset for faster lookup
category_dict = {row['wiki_id']: row['categories'] for row in wiki_id_categories}
print("Category dictionary created successfully.", "Length: ", len(category_dict))
print("Category dictionary: ", list(category_dict.items())[:10])

# Define a function to add categories to each row
def add_categories(example):
    example['categories'] = ', '.join(category_dict.get(example['wiki_id'], []))
    # print(f"Categories added for wiki_id {example['wiki_id']}", "Categories: ", example['categories'])
    return example

# Apply the function to the dataset
wiki_all_label = wiki_all_label.map(add_categories)
print("Categories added to the dataset successfully.")

# Save the updated dataset to a new file
wiki_all_label.save_to_disk(output_file)
print(f"Joined dataset saved to {output_file}")

# Print some information about the joined dataset
print(">>Total rows in joined dataset: ", len(wiki_all_label))
print(">>Total columns in joined dataset: ", wiki_all_label.column_names)