from mods.build_index import GovtSchemeDB

if __name__ == "__main__":
    db = GovtSchemeDB()

    # Usage
    results = db.search_schemes("crop insurance Karnataka", top_k=3)
    for r in results:
        print(f"{r['title']} - Distance: {r['distance']:.3f} - Description: {r['description'][:100]}...")

    db.close()