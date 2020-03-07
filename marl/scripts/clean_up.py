import os 
import sys 
import shutil 
import argparse 
import json 
import yaml 



def parse_args():
    parser = argparse.ArgumentParser("MADDPG with OpenAI MPE")
    parser.add_argument("-m", "--mode", type=str, default="remove",
                        help="mode of clean up")
    parser.add_argument("-s", "--source", type=str, default="scripts/TO_DELETE.txt",
                        help="file whose content are file paths to remove")    
    parser.add_argument("-p", "--prefix", type=str, default="ray_results",
                        help="prefix to file paths")
    return parser.parse_args()
    

def main(args):
    # remove folders
    if args.mode == "remove" and args.source is not None:
        if not os.path.exists(args.source):
            return
        content = read_file(args.source)
        unsuccessful = []
        # remove each file/folder
        for f_path in content:
            full_path = os.path.join(args.prefix, f_path)
            if not os.path.exists(full_path):
                continue
            try:
                remove(full_path)
                print("removed: ", full_path)
            except:
                unsuccessful.append(f_path)
                print("failed: ", full_path)
        # clear to_delete file
        f = open(args.source, 'w')
        for path in unsuccessful:
            f.write(path + "\n")
        f.close()



def eval_token(token):
    """ convert string token to int, float or str """ 
    if token.isnumeric():
        return int(token)
    try:
        return float(token)
    except:
        return token

def read_file(file_path, sep=","):
    """ read a file (json, yaml, csv, txt)
    """
    if len(file_path) < 1 or not os.path.exists(file_path):
        return None 
    # load file 
    f = open(file_path, "r")
    if "json" in file_path:
        data = json.load(f)
    elif "yaml" in file_path:
        data = yaml.load(f)
    else:
        sep = sep if "csv" in file_path else " "
        data = []
        for line in f.readlines():
            line_post = [eval_token(t) for t in line.strip().split(sep)]
            if len(line_post) == 1:
                line_post = line_post[0]
            if len(line_post) > 0:
                data.append(line_post)
    f.close()
    return data


def remove(*paths):
    """ remove list of files/folders """
    for path in paths:
        if not os.path.exists(path):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            raise ValueError("File type is not supported...")



if __name__ == '__main__':
    args = parse_args()
    main(args)

