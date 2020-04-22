# Starter code for CS 165B HW2 Spring 2019
def dot_product(a,b):
    if(len(a)!=len(b)):
        print("The length need to be the same")
    else:
        sum = 0
        for i in range(len(a)):
            sum+= a[i]*b[i]
        return sum

def get_w(a,b):
    w = []
    diff = []
    for i in range(len(a)):
        tem1 = a[i] - b[i]
        w.append(tem1)
    return w
    
def get_t(a,b):
    w = []
    diff = []
    for i in range(len(a)):
        tem1 = a[i] - b[i]
        tem2 = float(a[i]+b[i])/2
        w.append(tem1)
        diff.append(tem2)
    t = dot_product(w,diff)
    return t
def get_cen(datas):
    dim = len(datas[0])
    cen = []
    for i in range(dim):
        sum = 0
        for data in datas:
            sum+=data[i]
        cen.append(sum/len(datas))
    return cen

def discriminate(a,b):
    cen_a = get_cen(a)
    cen_b = get_cen(b)
    return get_w(cen_a,cen_b)



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
    result = {}
    train_anum = training_input[0][1]
    train_bnum = training_input[0][2]

    train_a = training_input[1:train_anum+1]
    train_b = training_input[train_anum+1:train_anum+train_bnum+1]
    train_c = training_input[train_anum+train_bnum+1:]

    dis_ab = discriminate(train_a,train_b)
    dis_ac = discriminate(train_a,train_c)
    dis_bc = discriminate(train_b,train_c)

    test_anum = testing_input[0][1]
    test_bnum = testing_input[0][2]
    test_cnum = testing_input[0][3]

    #for i in range(1,len(testing_input)):
    #    testing_input[i].append(1)
    t1 = get_cen(train_a)
    t2 = get_cen(train_b)
    t3 = get_cen(train_c)
    t_ab = get_t(t1,t2)
    t_ac = get_t(t1,t3)
    t_bc = get_t(t2,t3)


    test_a = testing_input[1:test_anum+1]
    test_b = testing_input[test_anum+1:test_anum+test_bnum+1]
    test_c = testing_input[test_anum+test_bnum+1:]

    TPA = 0
    FPA = 0
    FNA = 0

    FNB = 0
    TPB = 0
    FPB = 0

    FNC = 0
    TPC = 0
    FPC = 0

    TNA  = 0
    TNB = 0
    TNC = 0

    for data in test_a:
        if(dot_product(dis_ab,data)>=t_ab):
            if(dot_product(dis_ac,data)>=t_ac):
                TPA+=1
                TNB+=1
                TNC+=1
            else:
                FNA+=1
                TNB+=1
                FPC+=1
        else:
            if(dot_product(dis_bc,data)>=t_bc):
                FNA+=1
                FPB+=1
                TNC+=1
            else:
                FNA+=1
                TNB+=1
                FPC+=1
    for data in test_b:
        if(dot_product(dis_ab,data)>=t_ab):
            if(dot_product(dis_ac,data)>=t_ac):
                FPA+=1
                FNB+=1
                TNC+=1
            else:
                TNA+=1
                FNB+=1
                FPC+=1
        else:
            if(dot_product(dis_bc,data)>=t_bc):
                TNA+=1
                TPB+=1
                TNC+=1
            else:
                TNA+=1
                FNB+=1
                FPC+=1
    for data in test_c:
        if(dot_product(dis_ab,data)>=t_ab):
            if(dot_product(dis_ac,data)>=t_ac):
                FPA+=1
                TNB+=1
                FNC+=1
            else:
                TNA+=1
                TNB+=1
                TPC+=1
        else:
            if(dot_product(dis_bc,data)>=t_bc):
                TNA+=1
                FPB+=1
                FNC+=1
            else:
                TNA+=1
                TNB+=1
                TPC+=1

    TPRA = float(TPA)/(TPA+FNA)
    TPRB = float(TPB)/(TPB+FNB)
    TPRC = float(TPC)/(TPC+FNC)
    TPR = (TPRA+TPRB+TPRC)/3

    FPRA = float(FPA)/(FPA+TNA)
    FPRB = float(FPB)/(FPB+TNB)
    FPRC = float(FPC)/(FPC+TNC)
    FPR = (FPRA+FPRB+FPRC)/3

    E_A = float(FPA+FNA)/(TPA+FPA+FNA+TNA)
    E_B = float(FPB+FNB)/(TPB+FPB+FNB+TNB)
    E_C = float(FPC+FNC)/(TPC+FPC+FNC+TNC)
    ER = (E_A+E_B+E_C)/3

    ACC_A = float(TPA+TNA)/(TPA+FPA+FNA+TNA)
    ACC_B = float(TPB+TNB)/(TPB+FPB+FNB+TNB)
    ACC_C = float(TPC+TNC)/(TPC+FPC+FNC+TNC)
    ACC = (ACC_A+ACC_B+ACC_C)/3

    P_A = float(TPA)/(TPA+FPA)
    P_B = float(TPB)/(TPB+FPB)
    P_C = float(TPC)/(TPC+FPC)
    PRE = (P_A+P_B+P_C)/3

    result["tpr"] = TPR
    result["fpr"] = FPR
    result["error_rate"] = ER
    result["accuracy"] = ACC
    result["precision"] = PRE

    return result

