import random
import ast

t = 0
weightdecay = 0.001
BetaRMS = 0.999
BetaMomentum = 0.9
mode = 0
e = 2.718281828459045235360287471352
epsilon = 0.000000001
train = 0
update = 0
weightCount = 0
biasCount = 0
weights=[]
ContentOfLayers = []
dataI = []
dataO = []

if input("load from Save.txt (only if in current directory) (y/n): ") == "y":
    global data
    with open("Save.txt", "r", encoding="utf-8") as f:
        data = f.readlines()

    ContentOfLayers = ast.literal_eval(data[0].strip())
    layers = len(ContentOfLayers)
    weights = ast.literal_eval(data[1].strip())
    bias = ast.literal_eval(data[2].strip())

    if input("Continue Training? (y/n)") == "y":
        train = 1
        dataI = ast.literal_eval(data[3].strip())
        dataO = ast.literal_eval(data[4].strip())
else:
    ContentOfLayers = []
    layers = int(input("Number of layers: "))
    for i in range(layers):
        ContentOfLayers.append(int(input(f"Layer {i}: ")))

    for i in range(len(ContentOfLayers)-1):
        layerweights= []
        for n in range(ContentOfLayers[i+1]):
            limit = (6 / (ContentOfLayers[i] + ContentOfLayers[i+1])) ** 0.5
            neuronweights = [random.uniform(-limit, limit) for _ in range(ContentOfLayers[i])]
            layerweights.append(neuronweights)    
        weights.append(layerweights)

    bias = [[0] * ContentOfLayers[i] for i in range(layers)]


LearningRate = float(input("LearningRate: "))

for i in range(layers - 1):
    weightCount += ContentOfLayers[i] * ContentOfLayers[i+1]

for i in range(1, layers):
    biasCount += ContentOfLayers[i]

gradientsW = [
    [
        [0.0 for _ in neuron_weights]
        for neuron_weights in layer_weights
    ]
    for layer_weights in weights
]
gradientsB = [[0] * ContentOfLayers[i] for i in range(layers)]

mW = [[[0.0 for _ in n] for n in layer] for layer in weights]
vW = [[[0.0 for _ in n] for n in layer] for layer in weights]
mB = [[0.0]*ContentOfLayers[i] for i in range(layers)]
vB = [[0.0]*ContentOfLayers[i] for i in range(layers)]

Values = [[0] * ContentOfLayers[i] for i in range(layers)]
errors = [[0]*ContentOfLayers[j] for j in range(layers)]   #delta
unactivatedValues = [[0] * ContentOfLayers[i] for i in range(layers)]
expectedOutput = [0]*ContentOfLayers[-1]

print("weightCount: ", weightCount)
print("inputs:", Values[0])

def tanh(x):
    e_x = e**x
    e_mx = e**(-x)
    return (e_x - e_mx)/(e_x + e_mx)

def ForwardPass():
    lokal = layers
    for lay in range(1, lokal):    #layer
        for i in range(ContentOfLayers[lay]):      #neuron im endlayer
            total = 0
            for j in range(ContentOfLayers[lay-1]):      #neuronrn im startlayer
                total += Values[lay-1][j]*weights[lay-1][i][j]
            total = total + bias[lay][i]
            unactivatedValues[lay][i] = total    
            total = tanh(total)
            Values[lay][i] = total
print(ContentOfLayers)

def LossCalc():
    total = 0
    for i in range(ContentOfLayers[-1]):
        total += (expectedOutput[i]-Values[-1][i])**2
    total = 0.5*total
    return total

def Backpropagation():
    i = layers-1
    j = 0
    while j < len(Values[i]):  #output
        errors[i][j] = (Values[i][j]-expectedOutput[j])*(1-(Values[i][j])**2)
        j+=1
    j = 0
    i-=1
    while i>=1:
        for j in range(len(Values[i])):
            sumerrors = 0
            for k in range(len(errors[i+1])):
                sumerrors += errors[i+1][k]*weights[i][k][j]
            errors[i][j] = (sumerrors)*(1-(Values[i][j])**2)
        i-=1
    SetWeights(batchsize)
    SetBias(batchsize)


def SetWeights(n):
    for a in range(layers-2,-1,-1):       #(-2 because -1 for indexstart at 0 and -1 for weightindex on the neuron on the layer before)
        for b in range(len(weights[a])):
            for c in range(len(weights[a][b])):
                gradientsW[a][b][c] += errors[a+1][b]*Values[a][c]
                if update == 1:
                    g = gradientsW[a][b][c] / n
                    mW[a][b][c] = BetaMomentum *mW[a][b][c] + (1-BetaMomentum)*g
                    mhat = mW[a][b][c]/(1 - BetaMomentum**t)
                    vW[a][b][c] = BetaRMS*vW[a][b][c] + (1-BetaRMS)*(g**2)
                    vhat = vW[a][b][c]/(1 - BetaRMS**t)
                    weights[a][b][c] = weights[a][b][c]-LearningRate*weightdecay*weights[a][b][c]
                    weights[a][b][c] = weights[a][b][c]-LearningRate*mhat/(vhat**(1/2)+epsilon)
                    gradientsW[a][b][c] = 0.0

def SetBias(n):
    for a in range(layers-1,0,-1):
        for b in range(len(bias[a])):
            gradientsB[a][b] += errors[a][b]
            if update == 1:
                g = gradientsB[a][b] / n
                mB[a][b] = BetaMomentum*mB[a][b] + (1-BetaMomentum)*g
                vB[a][b] = BetaRMS*vB[a][b] + (1-BetaRMS)*(g**2)
                mhat = mB[a][b]/(1 - BetaMomentum**t)
                vhat = vB[a][b]/(1 - BetaRMS**t)
                bias[a][b] = bias[a][b]-LearningRate*mhat/(vhat**(1/2)+epsilon)
                gradientsB[a][b] = 0.0

def trainingMode():
    if train == 0:
        global mode, dataI, dataO, batchsize
        mode =int(input("Multiple Values?"))
        if mode == 0:
            for i in range(len(Values[0])):
                Values[0][i] = float(input(f"Input {i}: "))
        elif mode == 1:
            batchsize = int(input("batchsize(sets per update (at least 1)): "))
            datasets = int(input("Datasets: "))
            dataI = [[0.0]*ContentOfLayers[0] for _ in range(datasets)]
            dataO = [[0.0]*ContentOfLayers[-1] for _ in range(datasets)]
            for i in range(datasets):
                for n in range(ContentOfLayers[0]):
                    dataI[i][n] = float(input(f"Dataset {i}, Input {n}: "))
                for j in range(ContentOfLayers[-1]): 
                    dataO[i][j] = float(input(f"Dataset {i}, Output {j}: "))
        else:
            print("not accepted value, try again...")
            trainingMode()
    else:
        batchsize = int(input("batchsize(sets per update (at least 1)): "))
        mode = 1
trainingMode()

t = 0
if mode == 0:
    update = 1
    batchsize = 1
    for i in range(ContentOfLayers[-1]):
        expectedOutput[i] = float(input(f"Expected output {i}: "))
    epochs = int(input("Epochs: "))
    for i in range(epochs):
        t += 1
        ForwardPass()
        loss = LossCalc()
        Backpropagation()

        if weightCount < 1500:
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss}")
        else:
            print(f"Epoch {i}, Loss: {loss}")
    ForwardPass()
    print(f"Values: {Values[-1]}, expected: {expectedOutput}, Loss {LossCalc()}")

elif mode == 1:
    epochs = int(input("Epochs: "))
    shuffedlist = list(range(len(dataI)))
    b = 0
    for u in range(epochs):
        random.shuffle(shuffedlist)
        epochloss = 0.0
        for d in shuffedlist:
            Values[0] = dataI[d][:] 
            expectedOutput = dataO[d][:]
            ForwardPass()
            loss = LossCalc()
            epochloss += loss
            b += 1
            if b >= batchsize:
                b = 0
                update = 1
                t+=1
            Backpropagation()
            update = 0
            
        if weightCount < 1500:
            if u % 100 == 0:
                print(f"Epoch {u}, Loss: {epochloss/len(shuffedlist)}")
        else:
            print(f"Epoch {u}, Loss: {epochloss/len(shuffedlist)}")

    if input("Testquestion (y/n): ") == "y":
        for o in range(ContentOfLayers[0]):
            Values[0][o] = float(input(f"Testinput {o}: "))
        ForwardPass()
        print(f"Output: {Values[-1]}")
        a = 0
        while a == 0:
            if input("another question? (y/n) ") == "y":
                for o in range(ContentOfLayers[0]):
                    Values[0][o] = float(input(f"Testinput {o}: "))
                ForwardPass()
                print(f"Output: {Values[-1]}")  
            else:
                a = 1

if input("SAVE?(y/n)(only y if file Save.txt is in current directory)") == "y":
    with open("Save.txt", "w", encoding = "utf-8") as f:
        f.write(f"{ContentOfLayers}\n")
        f.write(f"{weights}\n")
        f.write(f"{bias}\n")
        f.write(f"{dataI}\n")
        f.write(f"{dataO}\n")
        
