import os

def make_folder(path, name):
    folder_path = os.path.join(path, name)
    if not folder_path:
        os.makedirs(folder_path)



if __name__ == "__main__":
    pass