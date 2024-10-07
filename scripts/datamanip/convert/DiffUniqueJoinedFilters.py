import sys

def main():
    base_unique_filters = set()
    with open ("e:\\data\\adsmf\\test\\bujf_nc.txt", encoding='utf-8', mode="r") as f:
        for line in f:
            base_unique_filters.add(line.strip().split('\t')[0])
    
    query_unique_filters = set()
    with open ("e:\\data\\adsmf\\test\\qujf_nc.txt", encoding='utf-8', mode="r") as f:
        for line in f:
            query_unique_filters.add(line.strip().split('\t')[0])
    
    missing = query_unique_filters.difference(base_unique_filters)
    print(f"Missing filters: {missing}")



if __name__ == "__main__":
    main()

    