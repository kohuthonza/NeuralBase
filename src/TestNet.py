import Net

def parse_args():
    print( ' '.join(sys.argv))

    parser = argparse.ArgumentParser(epilog="NeuralBase")
    parser.add_argument('-n', '--net',
                        type=str,
                        required=True,
                        help='Serializated net')
    parser.add_argument('-t', '--train',
                        type=str,
                        required=True,
                        help='JSON specification of train process')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='Name of net to be saved')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

if __name__ == "__main__":
    main()
