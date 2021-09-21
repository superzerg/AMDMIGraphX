import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for MIGraphX ROCTX Markers")
    parser.add_argument('--json_path',
                        type=str,
                        metavar='json_path',
                        help='path to json file')
    parser.add_argument('--parse', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')

    args = parser.parse_args()
    return args


def parse(file):
    with open(file, "r") as read_file:
        data = json.load(read_file)

    #Get marker names
    list_names = []
    for i in data:
        if (i):
            if("Marker start:" in i['name']) and (i['name'] not in list_names):
                list_names.append(i['name'])

    # Get timing information for each marker name
    print(list_names)
    list_times_per_names = []
    for name in list_names:
        print(name)
        temp_list = []
        for entry in data:
            if (entry) and (name == entry['name']):
                if(("gpu::" in name) and ("UserMarker frame:" in entry['args']['desc'])): #gpu side information
                    print(entry)
                    temp_list.append(int(entry.get('dur')))
                elif(("gpu::" not in name) and ("Marker start:" in entry['args']['desc'])): #cpu side information
                    print(entry)
                    temp_list.append(int(entry.get('dur')))
        list_times_per_names.append(temp_list)

    print(list_names)
    print(list_times_per_names)

    # Sum duration for each entry for a given name
    sum_per_name = []
    for list in list_times_per_names:
        sum_per_name.append(sum(list))

    print(sum_per_name)
    dictionary = dict(zip(list_names, sum_per_name))
    dictionary_sorted = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    print(dictionary)
    print(dictionary_sorted)
    #pprint.pprint(dictionary_sorted)
    total_time = sum(sum_per_name)

    print("\t ---- SUMMARY ----")
    for item in dictionary_sorted:
        print("%d us\t:\t%s" % (dictionary_sorted[item], item))
    print("TOTAL TIME: %s us"%total_time)

def main():
    args = parse_args()
    print(args)
    file = args.json_path

    if(args.run):
        run()
    
    if(args.parse):
        if not (file):
            raise Exception("JSON path is not provided for parsing.")
        parse(file)
    



if __name__ == "__main__":
    main()
