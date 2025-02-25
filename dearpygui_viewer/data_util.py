def transform_into_data_dict(slice):

    sumDeltaTime = 0.0
    dict = {}
    for frame in slice:
        sumDeltaTime += frame.secondsSinceLastUpdate
        for key, value in frame.data.items():
            if key not in dict:
                dict[key] = []
            dict[key].append(value)

    return dict, sumDeltaTime

def print_info_from_graph_dict(dict: dict):
    if(dict == None or len(dict) == 0):
        return

def get_graphs_and_deltaTime_from_slice(slice):

    dict, deltaTime = transform_into_data_dict(slice)
    print_info_from_graph_dict(dict)
    return dict, deltaTime


def split_by_identifierGroups(dataDict, groups, exclude="") -> list:
    ArraysOfDataPerGroup = [[] for _ in range(len(dataDict))] #why bnot len(groups)
    dictIndex = {}
    for group in groups:
        for key, value in dataDict.items():
            # compare key against all elements in group
            for identifier in group:
                if identifier in key  and exclude not in key:
                    if key not in dictIndex:
                        dictIndex[key] = len(dictIndex)
                    ArraysOfDataPerGroup[dictIndex[key]].append(value)
    return ArraysOfDataPerGroup