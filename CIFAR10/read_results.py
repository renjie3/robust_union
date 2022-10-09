import argparse

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--file', default='', type=str)

# args parse
args = parser.parse_args()

# f = open('./logfile/{}.log'.format(args.file), 'r')
# lines = f.readlines()
# f.close()

f = open('./input.txt'.format(args.file), 'r')
lines = f.readlines()
f.close()

print_list = []
loss_print_list = []
for line in lines:
    splits = line.strip().split(':')
    if len(splits) > 1 and 'Alpha' in line:
        result = splits[-1]
        print_list.append(result)
        if 'Alpha: 10.0000' in line:
            print_list.append('\n')

    if len(splits) > 1 and 'Test Acc Clean' in line:
        result = splits[-1]
        print_list.append(result)
        print_list.append('\n')

    # print(splits)

    loss_print_list.append(splits[2].replace(' Acc', ''))

    
    
print('\t'.join(loss_print_list))
