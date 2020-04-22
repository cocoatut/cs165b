# Starter code for CS 165B HW2 Spring 2019

def centroid(data_set):
    centroid_point = []
    dimension = len(data_set[0])
    point_sum = 0
    result = 0
    for d in range(dimension):
        point_sum = 0
        result = 0
        for point in data_set:
            point_sum += point[d]
        result = point_sum / len(data_set)
        centroid_point.append(result)
    return centroid_point

def dot_product(point1, point2):
    if(len(point1) != len(point2)):
        print("Error")
    else:
        result = 0
        for i in range(len(point1)):
            result += point1[i] * point2[i]
        return result

def get_w(point1, point2):
    w = []
    for i in range(len(point1)):
        w.append(point1[i] - point2[i])
    return w

def get_t(point1, point2):
    t = 0
    part1 = []
    part2 = []
    for i in range(len(point1)):
        part1.append((point1[i] + point2[i]) / 2)
        part2.append(point1[i] - point2[i])
    t = dot_product(part1, part2)
    return t

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """
    # training class: get A/B/C length
    training_A_length = training_input[0][1]
    training_B_length = training_input[0][2]
    training_C_length = training_input[0][3]
    # get data for A/B/C
    training_A_data = training_input[1 :training_A_length + 1]
    training_B_data = training_input[training_A_length + 1 : training_A_length + training_B_length + 1]
    training_C_data = training_input[training_A_length + training_B_length + 1 : training_A_length + training_B_length + training_C_length + 1]
    centroid_A = centroid(training_A_data)
    centroid_B = centroid(training_B_data)
    centroid_C = centroid(training_C_data)


    # testing class: get A/B/C length
    testing_A_length = testing_input[0][1]
    testing_B_length = testing_input[0][2]
    testing_C_length = testing_input[0][3]
    # get data for A/B/C
    testing_A_data = testing_input[1 :testing_A_length + 1]
    testing_B_data = testing_input[testing_A_length + 1 : testing_A_length + testing_B_length + 1]
    testing_C_data = testing_input[testing_A_length + testing_B_length + 1 : testing_A_length + testing_B_length + testing_C_length + 1]

    TP_A = 0
    TP_B = 0
    TP_C = 0
    FP_A = 0
    FP_B = 0
    FP_C = 0
    TN_A = 0
    TN_B = 0
    TN_C = 0
    FN_A = 0
    FN_B = 0
    FN_C = 0
    t_AB = get_t(centroid_A, centroid_B)
    t_AC = get_t(centroid_A, centroid_C)
    t_BC = get_t(centroid_B, centroid_C)
    w_AB = get_w(centroid_A, centroid_B)
    w_AC = get_w(centroid_A, centroid_C)
    w_BC = get_w(centroid_B, centroid_C)

    # class A
    for point in testing_A_data:
        if dot_product(w_AB, point) > t_AB:
            if dot_product(w_AC, point) > t_AC:
                TP_A += 1
                TN_B += 1
                TN_C += 1
            else:
                FN_A += 1
                TN_B += 1
                FP_C += 1
        else:
            if dot_product(w_BC, point) > t_BC:
                FN_A += 1
                FP_B += 1
                TN_C += 1
            else:
                FN_A += 1
                TN_B += 1
                FP_C += 1
    
    # class B
    for point in testing_B_data:
        if dot_product(w_AB, point) > t_AB:
            if dot_product(w_AC, point) > t_AC:
                FP_A += 1
                FN_B += 1
                TN_C += 1
            else:
                TN_A += 1
                FN_B += 1
                FP_C += 1
        else:
            if dot_product(w_BC, point) > t_BC:
                TN_A += 1
                TP_B += 1
                TN_C += 1
            else:
                TN_A += 1
                FN_B += 1
                FP_C += 1
    # class C
    for point in testing_C_data:
        if dot_product(w_AB, point) > t_AB:
            if dot_product(w_AC, point) > t_AC:
                FP_A += 1
                TN_B += 1
                FN_C += 1
            else:
                TN_A += 1
                TN_B += 1
                TP_C += 1
        else:
            if dot_product(w_BC, point) > t_BC:
                TN_A += 1
                FP_B += 1
                FN_C += 1
            else:
                TN_A += 1
                TN_B += 1
                TP_C += 1
    
    # true postive rate = TP / P (P = TP + FN)
    TPR_A = float(TP_A) / (TP_A + FN_A)
    TPR_B = float(TP_B) / (TP_B + FN_B)
    TPR_C = float(TP_C) / (TP_C + FN_C)
    TPR = (TPR_A + TPR_B + TPR_C) / 3

    # false postive rate = FP / N (N = FP + TN)
    FPR_A = float(FP_A) / (FP_A + TN_A)
    FPR_B = float(FP_B) / (FP_B + TN_B)
    FPR_C = float(FP_C) / (FP_C + TN_C)
    FPR = (FPR_A + FPR_B + FPR_C) / 3

    # error rate = (FP + FN) / (P + N)
    error_rate_A = float(FP_A + FN_A) / (TP_A + FN_A + FP_A + TN_A)
    error_rate_B = float(FP_B + FN_B) / (TP_B + FN_B + FP_B + TN_B)
    error_rate_C = float(FP_C + FN_C) / (TP_C + FN_C + FP_C + TN_C)
    error_rate = (error_rate_A + error_rate_B + error_rate_C) / 3

    # accuracy = (TP + TN) / (P + N)
    accuracy_A = float(TP_A + TN_A) / (TP_A + FN_A + FP_A + TN_A)
    accuracy_B = float(TP_B + TN_B) / (TP_B + FN_B + FP_B + TN_B)
    accuracy_C = float(TP_C + TN_C) / (TP_C + FN_C + FP_C + TN_C)
    accuracy = (accuracy_A + accuracy_B + accuracy_C) / 3

    # precision = TP / P_estimated
    precision_A = float(TP_A) / (TP_A + FP_A)
    precision_B = float(TP_B) / (TP_B + FP_B)
    precision_C = float(TP_C) / (TP_C + FP_C)
    precision = (precision_A + precision_B + precision_C) / 3

    result = {}
    result["fpr"] = FPR
    result["tpr"] = TPR
    result["error_rate"] = error_rate
    result["accuracy"] = accuracy
    result["precision"] = precision

    return result
