"""This script extracts categories from a Wikipedia dump for specific wiki IDs. 
The wiki IDs are provided in a text file, and the categories are extracted from the Wikipedia dump for these IDs. 
The categories are then saved as an Arrow dataset for further processing.
More info: https://microsoftapc-my.sharepoint.com/:w:/g/personal/t-asutradhar_microsoft_com/Ec7UTAvWVhpKvBiDSF8fLIcBLtwTdKzqb7_L-lD2HSNsGw?e=TeLmgB"""



import mwxml
from datasets import Dataset
import pandas as pd
import time
from tqdm import tqdm
from bs4 import XMLParsedAsHTMLWarning
import warnings
import re

"""constants"""
page_count = 0

# Filter out warnings related to parsing XML as HTML
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def create_page_id_map(dump_file, limit=None):
    """Create a map of page IDs to page objects."""
    print(f"[{time.strftime('%H:%M:%S')}] Creating page ID map from dump file: {dump_file}")
    dump = mwxml.Dump.from_file(open(dump_file, 'rb')).pages
    page_id_map = {}
    count = 0

    for page in tqdm(dump, desc="Creating page ID map"):
        if limit and count >= limit:
            break
        page_id_map[page.id] = page
        count += 1

    return page_id_map

def count_articles_in_dump(dump_file):
    """Count the number of articles in the Wikipedia dump."""
    print(f"[{time.strftime('%H:%M:%S')}] Loading dump file: {dump_file}")
    dump = mwxml.Dump.from_file(open(dump_file, 'rb')).pages
    print(f"[{time.strftime('%H:%M:%S')}] Counting articles in dump: {dump_file}")
    article_count = 0

    for _ in tqdm(dump, desc="Counting articles"):
        article_count += 1

    return article_count

def extract_categories_from_dump(dump_file, wiki_ids, limit=None, bad_ids_file = "bad_ids3.txt"):
    """Extract categories from Wikipedia dump for specific wiki IDs."""
    global page_count  # Declare page_count as global
    print(f"[{time.strftime('%H:%M:%S')}] Loading dump file: {dump_file}")
    dump = mwxml.Dump.from_file(open(dump_file, 'rb')).pages
    print(f"[{time.strftime('%H:%M:%S')}] Extracting categories from dump: {dump_file}")
    wiki_id_categories = {}
    

    category_pattern = re.compile(r'\[\[Category:(.*?)\]\]')
    wiki_ids_set = set(map(str, wiki_ids))

    for page in tqdm(dump, desc="Processing pages"):
        if limit and page_count >= limit:
            break
        if str(page.id) in wiki_ids_set:

            for revision in page:
                if revision.text:
                    categories = category_pattern.findall(revision.text)
                    if categories:
                        if len(categories) > 10:
                            categories = sorted(categories, key=lambda x: len(x.split()))[:10]
                            categories = [cat for cat in categories if len(cat.split()) < 4]
                        wiki_id_categories[page.id] = categories
                        # print(f"Page ID: {page.id}, Categories: {categories}")
                    else:
                        # print(f"Page ID: {page.id}, Link: https://en.wikipedia.org/wiki?curid={page.id}")
                        with open(bad_ids_file, 'a', encoding='utf-8') as file:
                            file.write(f"{page.id}, ")
                            file.write(f"https://en.wikipedia.org/wiki?curid={page.id}\n")

        # else:
            # print(f"Page ID: {page.id}, Not in wiki_ids")
        page_count += 1

    return wiki_id_categories

def save_categories_arrow(wiki_id_categories, output_file):
    df = pd.DataFrame(list(wiki_id_categories.items()), columns=['wiki_id', 'categories'])
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(output_file)  # Save as Arrow file

def load_wiki_ids(file_path):
    with open(file_path, 'r') as file:
        wiki_ids = [line.strip() for line in file.readlines()]
    return wiki_ids

def main():
    dump_file = 'enwiki-20250201-pages-articles-multistream.xml'  # Replace with the path to your Wikipedia dump file
    wiki_ids_file = 'wiki_ids.txt'  # Replace with the path to your file containing wiki IDs
    # limit = 10000

    print(f"[{time.strftime('%H:%M:%S')}]Started extracting categories from dump...")
    start_time = time.time()

    # Load the list of wiki IDs
    wiki_ids = load_wiki_ids(wiki_ids_file)
    wiki_ids = sorted(wiki_ids)
    print(f"Total number of wiki IDs: {len(wiki_ids)}")

    # Extract categories for the specific wiki IDs
    wiki_id_categories = extract_categories_from_dump(dump_file, wiki_ids)

    print(f"[{time.strftime('%H:%M:%S')}]Completed in {time.time() - start_time:.2f} sec.")

    # Save as Arrow dataset
    current_date_time = time.strftime('%Y%m%d%H%M%S')
    output_file = f'wiki_id_categories_from_dump_{current_date_time}.arrow'
    start_time = time.time()
    save_categories_arrow(wiki_id_categories, output_file)
    print(f"Saved in {time.time() - start_time:.2f} sec.")

if __name__ == "__main__":
    main()