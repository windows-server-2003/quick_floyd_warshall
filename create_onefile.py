import os 
script_path = os.path.dirname(os.path.realpath(__file__))

def load_cpp_with_include_expansion(fname) :
	res = ""
	with open(script_path + "/" + fname) as f:
		for line in f.readlines() :
			if line.startswith("#pragma once") : continue
			if line.startswith("#include \"") :
				header_name = line.split('\"')[1]
				ok = False
				for base in ['.', os.path.dirname(fname)] :
					cur_header_name = base + '/' + header_name
					if os.path.isfile(cur_header_name) :
						res += load_cpp_with_include_expansion(cur_header_name) + "\n"
						ok = True
						break
				if not ok :
					print('header not found: ' + header_name)
					exit(1)
			else : res += line
	return res

res_code = load_cpp_with_include_expansion("quick_floyd_warshall/qfw.h")

"""
res = \
	"#pragma GCC target(\"avx512f,avx512dq,avx512bw\")\n" + \
	"#pragma GCC optimize(\"O3,unroll-loops\")\n" + res
"""

with open(script_path + "/combined.h", "w") as f :
	f.write(res_code)
