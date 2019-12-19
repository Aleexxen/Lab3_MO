import math

u = [4, 3]
eps = 0.000001

def fun(x):     #((x * x + y - 11) ** 2 + (x + y * y - 7) ** 2)
    return (x[0] * x[0] + x[1] - 11) ** 2 + (x[0] + x[1] * x[1] - 7) ** 2

def grad(x):
    #return [2 * (2 * x[0] * (x[0] * x[0] + x[1] - 11) + x[0] + x[1] * x[1] - 7), 2 * (x[0] * x[0] + 2 * x[1] * (x[0] + x[1] * x[1] - 7) + x[1] - 11)]
    delta = 1e-6
    ans = []
    orig = fun(x)
    for i in range(len(x)):
        x[i] += delta
        ans.append((fun(x) - orig) / delta)
        x[i] -= delta
    return ans

def hessian(x):
    #return [[12 * x[0] * x[0] + 4 * x[1] - 42, 4 * x[0] + 4 * x[1]], [4 * x[0] + 4 * x[1], 4 * x[0] + 12 * x[1] * x[1] - 26]]
    gr = grad(x)
    delta = 1e-4
    res = []
    for i in range(len(gr)):
        orig = gr[i]
        ans = []
        for j in range(len(x)):
            x[j] += delta
            ans.append((grad(x)[i] - orig) / delta)
            x[j] -= delta
        res.append(ans)
    return res

def mod(x):
    res = 0;
    for i in range(len(x)):
        res += x[i] * x[i]
    #print(res)
    return math.sqrt(res)

def inverse(x):
    a, b, c, d = x[0][0], x[0][1], x[1][0], x[1][1]
    return [[d/(-b * c + a * d), b/(b * c - a * d)], [c/(b * c - a * d), a/(-b * c + a * d)]]

def mulMat(a, b):
    return [a[0][0] * b[0] + a[0][1] * b[1], a[1][0] * b[0] + a[1][1] * b[1]]

def newton():
    k = 0
    x = u
    gr = grad(x)
    while (mod(gr) > eps):
        k += 1
        h = hessian(x)
        a = mulMat(inverse(h), gr)
        #print(a)
        for i in range(len(x)):
            x[i] -= a[i]
        gr = grad(x)
    print(x)
    print(fun(x))

#print(grad([10, 5]))
#print(hessian([10, 5]))
newton()
