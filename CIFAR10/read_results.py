import argparse

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--file', default='', type=str)

# args parse
args = parser.parse_args()

f = open('./logfile/{}.log'.format(args.file), 'r')
lines = f.readlines()
f.close()

print_list = []
for line in lines:
    splits = line.strip().split(': ')
    if len(splits) > 1 and 'Alpha' in line:
        result = splits[-1]
        print_list.append(result)
    
print('\t'.join(print_list))
