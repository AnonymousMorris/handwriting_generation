import pickle as pkl

path = "./data/0/0.pkl"

if __name__ == "__main__":
    with open(path, 'rb') as f:
        raw_data = pkl.load(f)
        print(raw_data)
        print()
        print(list(raw_data.keys()))
        drawing = raw_data['drawing']
        print(f"Type: {type(drawing)}")
        print(f"Length: {len(drawing)}")
        print("First item:", drawing[0])
        print("Len of First item:", len(drawing[0]))

