import numpy as np
import re
import sys


data = np.genfromtxt(str(sys.argv[1]), unpack=True)
def columnsettings(a):
    lengthnv = 0
    lnv = 0
    vz = ""
    lengthnn = 0
    lnn = 0
    lenghtu = 0
    lu = 0
    lenghte = 0
    le = 0

    num = re.compile(r"^-{0,1}\d+(.\d+(\\pm\d+(.\d+){0,1}(e\(-{0,1}\d+\)){0,1}){0,1}){0,1}$")
    v = re.compile(r"[-+]")
    k = re.compile(r".")
    nv = re.compile(r"(?=^-{0,1})\d+")
    nn = re.compile(r"(?=^-{0,1}\d+.)\d+")
    u = re.compile(r"(?=^-{0,1}\d+(.\d+){0,1}\\pm)\d+(.\d+){0,1}")
    e = re.compile(r"(?=^-{0,1}\d+(.\d+(\\pm\d+(.\d+){0,1}){0,1}){0,1}e)\d+")
#    e = re.compile(r"(?=^-{0,1}\d+(.\d+(\\pm\d+(.\d+){0,1}e\()-{0,1}\d+")

    for i in a:
        if not num.match(str(i)):
            return "c"
        c = nv.match(str(i))
        lengthnv = len(a.group()) if a else 0
        a = nn.match(str(i))
        lengthnn = len(a.group()) if a else 0
        b = u.match(str(i))
        lengthu = len(b.group()) if b else 0
        a = e.match(str(i))
        lengthe = len(a.group()) if a else 0
        if(c):
            if(v.match(b.group())): vz = "+"
        if(b):
            if(k.match(b.group())): lengthu -= 1
        if(lengthnv > lnv): lnv = lengthnv
        if(lengthnn > lnn): lnn = lengthnn
        if(lengthu > lu): lu = lengthu
        if(lengthe > le): le = lengthe
    return "S[table-format=" + vz + str(lnv) + "." + str(lnn-1) + "(" + str(lu) + ")e" + str(le) + "]"

file = open(str(sys.argv[1]), "r")
caption = file.readline()
kopfzeile = file.readline()
file.close()

out = open("build" + str(sys.argv[1])[7:-3] + "tex", "w")
out.write("\\begin{table}\n")
out.write("\\label{tab:" + str(sys.argv[1])[7:-3] + "}\n")
out.write("\t\\centering\n")
out.write("\t\\caption{")
if((caption[0] == "#") & (caption[1] == "#")):
    out.write(caption[2:])
else:
    kopfzeile = caption
out.write(".}\n")
out.write("\t\\begin{tabular}{")
for i in data:
    out.write(columnsettings(i) + " ")
out.write("}\n")
out.write("\t\t\\toprule\n")
if(kopfzeile[0] == "#"):
    kopfzeile = [x.strip() for x in kopfzeile[1:].split(',')]
    out.write("\t")
#   for i in range(len(kopfzeile) -1):
    for i in range(min(len(kopfzeile), int(data.size/data[0].size))):
        out.write("\t{$" + str(kopfzeile[i]) + "$} & ")
    out.write("\t{$" + str(kopfzeile[-1]) + "$} \\\\\n")
else:
    for i in ragne(int(data.size/data[0].size)-1):
        out.write("\t{$$} &")
    out.write("\t{$$} \\\\\n")

out.write("\t\t\\midrule\n")

for j in range(data[0].size):
    out.write("\t\t")
    for i in range(int(round(data.size/data[0].size)) -1):
        out.write(str(data[i][j]) + "\t& ")
    out.write(str(data[-1][j]) + "\t\\\\\n")

out.write("\t\t\\bottomrule\n")
out.write("\t\\end{tabular}\n")
out.write("\\end{table}\n")

