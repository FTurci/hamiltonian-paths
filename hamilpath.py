import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
from lempel_ziv_complexity import lempel_ziv_complexity
from copy import deepcopy
import zlib 

class HamiltonianPath:
	def __init__(self,n=None, path=None):
		
		if path is None:
			self.n = n
			self.G = nx.Graph()
		else:
			self.G = path.G.copy()
			self.n = path.n
			self.head = path.head
			self.tail = path.tail

	@property
	def sequence(self):
		return nx.shortest_path(self.G,self.head, self.tail)

	def draw(self,node_size=100):
		pos=nx.get_node_attributes(self.G,'pos')

		nx.draw(self.G,pos,alpha=1.0,node_shape='s',node_size=node_size)

	def backbite(self):
		# from http://datagenetics.com/blog/december22018/index.html
		# 1 - Select either of the two end points on the current graph.
		n = self.n
		if np.random.random()<0.5:
			s = self.head
			other = self.tail
			selection = 'head'
		else:
			s  = self.tail
			other = self.head
			selection = 'tail'

		# print ("head, tail", self.head, self.tail)
		assert self.G.degree[s]==1, "The selected node is not an endpoint"

		i,j = np.unravel_index(s,(n, n))

		
		linked_nodes = list(list(self.G.edges(s))[0])
		linked_nodes.remove(s)
		linked_node = linked_nodes[0]
		# print("  Selected",s, selection, "with linked node", linked_node, "ij", i,j, "with other", other)

		if i==0:
			if j>0 and j<n-1:
				neighs  = [s-1, s+1,s+n]
				
			elif j==0:
				neighs = [s+1, s+n]

			elif j==n-1:
				neighs = [s-1, s+n]

		elif i==n-1:

			if j>0 and j<n-1:
				neighs  = [s-1, s+1,s-n]
				
			elif j==0:
				neighs = [s+1, s-n]

			elif j==n-1:
				neighs = [s-1, s-n]

		else:
			if  j==0:
				neighs = [s-n, s+1, s+n]
			elif j==n-1:
				neighs = [s-1, s-n, s+n]
			else:
				neighs = [s-1, s+1, s-n, s+n]


		neighs.remove(linked_node)
		
		# print("neighs", neighs)
		# pick a neighbour
		pick = np.random.choice(neighs)
		# print("picked", pick)
		if pick == other:
			# connecting to other endpoint, so we first remove the edgeof the pther endpoint

			linked_nodes = list(list(self.G.edges(other))[0])
			linked_nodes.remove(other)
			linked_node = linked_nodes[0]
			self.G.remove_edge(other, linked_node)
			# ... and add an edge between s and other

			self.G.add_edge(s,other)

			# print("edges at linked_node", self.G.edges(linked_node),other)
			# print("edges at other", self.G.edges(other))

			if selection==0 :
				self.head = linked_node 
			elif  selection==1:
				self.tail = linked_node

		else:

			original_edges  = list(self.G.edges(pick))
			# print(original_edges,pick)
			# form loop
			self.G.add_edge(s,pick)

			path_to_other = nx.shortest_path(self.G,pick,other)
			# print(original_edges)
			for edge in original_edges: 
				if (path_to_other[0], path_to_other[1]) == edge:
					# keep edge
					pass
				else:
					# cut edge
					destination = edge[1]
					self.G.remove_edge(edge[0], edge[1])


			if selection=='head':
				self.head = destination 
			elif  selection=='tail':
				self.tail = destination


class Serpentine(HamiltonianPath):
	def __init__(self,n):
		self.n = n
		self.G = nx.Graph()
	

		for k in range(0,n*n):
			i,j =  np.unravel_index(k,(n,n))
			self.G.add_node(k, pos=(i,j))

		for k in range(0,n*n-1):
			i,j =  np.unravel_index(k,(n,n))
			if  (k+1)%n!=0:
				self.G.add_edge(k,k+1)
			elif i%2==0:
				self.G.add_edge(k,k+n)
			if k%n==0 and i%2==1 and k+n<n*n:
				self.G.add_edge(k,k+n)
		
		if n%2 == 1:
			self.head = 0
			self.tail = n*n-1
		else:
			self.head = 0
			self.tail = n*n-n




def evaluate(image, path):
	seq = path.sequence
	idx = np.unravel_index(seq,(path.n, path.n))
	# return "".join(image[idx].astype(int).astype(str))
	return bytes(str.encode("".join((image[idx].astype(int)).astype(str))))

def compression_length():
	pass
# setup grid of points
n=20

x = np.linspace(0,20,10)
print(x)
X,Y = np.meshgrid(x,x)

points = np.array([X.flatten(),Y.flatten()]).T

nr = 60
# points = np.vstack((
	# points,
points = 	np.array([np.random.uniform(0,20, nr), np.random.uniform(0,20, nr) ]).T 
	# )
	# )
H,edg = np.histogramdd(points, bins=(n,n))

# H = H>0
H = np.zeros(H.shape)
H[12,5:15]=1
# H[3,5:16]=1
# H[5:10,5]=1
H[5:13,15]=1
# plt.scatter(points[:,1], points[:,0])
plt.imshow(H.T, cmap=plt.cm.summer)

# plt.show()
S = Serpentine(n=n)

original =evaluate(H,S)

def information(code):
	compressed = zlib.compress(code, level=9)
	return len(compressed)#/len(code)

# print(len(original), lempel_ziv_complexity(original))


# S.draw(node_size=1)
# # plt.show()
# # print(S.sequence)


# warmup the snake
# for k in range(10000):
	# S.backbite()


# subsamples = 3
# for k in range(100000):
# 	# explore several states seperately
# 	lzo = lempel_ziv_complexity(evaluate(H,S))
# 	copies ,lzs= [S],[lzo]
# 	for sub in range(subsamples):
# 		Scopy = HamiltonianPath(path=S)
# 		for u in range(np.random.binomial(20,0.5)):
# 			Scopy.backbite()
# 		lz = lempel_ziv_complexity(evaluate(H,Scopy))
# 		copies.append(Scopy)
# 		lzs.append(lz)
# 	optimum = np.argmin(lzs)
# 	S = copies[optimum]
# 	print(lzs[optimum], lzs)	

# try with some sort of simulated annealing
beta0 = 0.2
for k in range(1000):
	beta = beta0*np.log((k+1)) #boltzmann annealing
	Scopy = HamiltonianPath(path=S)
	lzold = information(evaluate(H,S))
	if k%1000==0:
		print(k, beta, lzold)
		plt.imshow(H.T, cmap=plt.cm.summer)
		S.draw(node_size=1)
		plt.axis('equal')
		# plt.show()
		plt.title(f"k:{k}, cost:{lzold}")
		plt.savefig("fig%07d.png"%k)
		plt.close()
	
	S.backbite()
	lznew = information(evaluate(H,S))
	# print(lzold)
	if (np.random.random() < np.exp(-beta*(lznew-lzold))):
		pass
		# print(k,"reject")
	else:
		# print(k,"accept")
		S = Scopy

	