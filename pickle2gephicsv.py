import sys
import assignment_analysis as asg
import pickle

def main(datapath, savepath):
    data = pickle.load(open(datapath,'rb'))
    asg.write_graph(savepath, G=data['G'])


if __name__ == '__main__':
    main(*sys.argv[1:])
