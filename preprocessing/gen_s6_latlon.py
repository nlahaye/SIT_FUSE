from misc_utils import dummyLatLonS6
from utils import read_yaml


def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)
    data = yml_conf["filenames"]

    dummyLatLonS6(data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)


