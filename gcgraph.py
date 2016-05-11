# GCGRAPH
# Maxflow
import ctypes

class Pointer:
	def __init__(self, var):
		self.id = id(var)

	def get_value(self):
		return ctypes.cast(self.id, ctypes.py_object).value


class Vertex:
	def __init__(self):
		self.next = 0 # Initialized and used in maxflow() only
		self.parent = 0
		self.first = 0
		self.ts = 0
		self.dist = 0
		self.weight = 0
		self.t = 0

class Edge:
	def __init__(self):
		self.dst = 0
		self.next = 0
		self.weight = 0.0

class GCGraph:
	def __init__(self, vertex_count, edge_count):
		self.vertexs = []
		self.edges = []
		self.flow = 0
		self.vertex_count = vertex_count
		self.edge_count = edge_count

	def add_vertex(self):
		v = Vertex()
		self.vertexs.append(v)
		return len(self.vertexs) -  1

	def add_edges(self, i, j, w, revw):

		a = len(self.edges)
		# As is said in the C++ code, if edges.size() == 0, then resize edges to 2.

		# if a == 0:
		# 	a = 2

		fromI = Edge()
		fromI.dst = j
		fromI.next = self.vertexs[i].first
		fromI.weight = w
		self.vertexs[i].first = a
		self.edges.append(fromI)

		toI = Edge()
		toI.dst = i
		toI.next = self.vertexs[j].first
		toI.weight = revw
		self.vertexs[j].first = a + 1
		self.edges.append(toI)



	def add_term_weights(self, i, source_weight, sink_weight):
		dw = self.vertexs[i].weight
		if dw > 0:
			source_weight += dw
		else:
			sink_weight -= dw
		self.flow += source_weight if source_weight < sink_weight else sink_weight
		self.vertexs[i].weight = source_weight - sink_weight

	def max_flow(self):
		TERMINAL = -1
		ORPHAN = -2
		stub = Vertex()
		nilNode = Pointer(stub)
		first = Pointer(stub)
		last = Pointer(stub)
		curr_ts = 0
		stub.next = nilNode.get_value()		
		# # print(first.get_value() == nilNode.get_value())
		
		orphans = []

		# initialize the active queue and the graph vertices
		for i in range(len(self.vertexs)):
			v = self.vertexs[i]
			v.ts = 0
			if v.weight != 0:
				last.get_value().next = v
				last.id = id(v)
				v.dist = 1
				v.parent = TERMINAL
				v.t = v.weight < 0
			else:
				v.parent = 0
			# # print(first.get_value().next == nilNode.get_value())
		first.id = id(first.get_value().next)
		last.get_value().next = nilNode.get_value()
		nilNode.get_value().next = 0
		# # print(first.get_value() == nilNode.get_value())

		# count = 0
		# Search Path -> Augment Graph -> Restore Trees
		while True:
			# print('1','\n', [x.t for x in self.vertexs])

			# count += 1
			# # print(count)
			e0 = -1
			ei = 0
			ej = 0

			while first.get_value() != nilNode.get_value():
				v = first.get_value()
				if v.parent:
					vt = v.t
					ei = v.first
					while ei != 0:
						if self.edges[ei^vt].weight == 0:
							ei = self.edges[ei].next
							continue	
						u = self.vertexs[self.edges[ei].dst]
						if not u.parent:
							u.t = vt
							u.parent = ei ^ 1
							u.ts = v.ts
							u.dist = v.dist + 1
							if not u.next:
								u.next = nilNode.get_value()
								last.get_value().next = u
								last.id = id(u)
							ei = self.edges[ei].next
							continue
						if u.t != vt:
							e0 = ei ^ vt
							break
						if u.dist > v.dist + 1 and u.ts <= v.ts:
							u.parent = ei ^ 1
							u.ts = v.ts
							u.dist = v.dist + 1
						# # print(self.edges[ei].next)
						ei = self.edges[ei].next
					if e0 > 0:
						break
				first.id = id(first.get_value().next)
				# first = first.next
				v.next = 0
			
			# print('2','\n', [x.t for x in self.vertexs])

			if e0 <= 0:
				break

			minWeight = self.edges[e0].weight
			for k in range(1, -1, -1):
				v = self.vertexs[self.edges[e0^k].dst]
				while True:
					# # print('f')
					ei = v.parent
					if ei < 0:
						break
					weight = self.edges[ei^k].weight
					minWeight = min(minWeight, weight)
					v = self.vertexs[self.edges[ei].dst]
				weight = abs(v.weight)
				minWeight = min(minWeight, weight)
			
			self.edges[e0].weight -= minWeight
			self.edges[e0^1].weight += minWeight
			self.flow += minWeight

			for k in range(1, -1, -1):
				v = self.vertexs[self.edges[e0^k].dst]
				while True:
					# # print('d')
					ei = v.parent
					if ei < 0:
						break
					self.edges[ei^(k^1)].weight += minWeight
					self.edges[ei^k].weight -= minWeight
					if self.edges[ei^k].weight == 0:
						orphans.append(v)
						v.parent = ORPHAN
					v = self.vertexs[self.edges[ei].dst]
				v.weight = v.weight + minWeight*(1-k*2)
				if v.weight == 0:
					orphans.append(v)
					v.parent = ORPHAN
			curr_ts += 1
			
			while len(orphans) != 0:
				# v2 = orphans[-1]
				# print('v', v2)
				v2 = orphans.pop()
				minDist = float('inf')
				e0 = 0
				vt = v2.t
				
				ei = v2.first
				bcount = 0
				while ei != 0:
					bcount += 1
					# print('1', bcount)
					# print(self.edges[ei^(vt^1)].weight)
					if self.edges[ei^(vt^1)].weight == 0:
						ei = self.edges[ei].next
						continue
					u = self.vertexs[self.edges[ei].dst]
					if u.t != vt or u.parent == 0:
						ei = self.edges[ei].next
						continue

					d = 0
					while True:
						# bcount += 1
						# print(bcount)
						if u.ts == curr_ts:
							d += u.dist
							break
						ej = u.parent
						d += 1
						# print(d)
						if ej < 0:
							if ej == ORPHAN:
								d = float('inf') - 1
							else:
								u.ts = curr_ts
								u.dist = 1
							break
						u = self.vertexs[self.edges[ej].dst]
					# print(ei)
						# print('u', u)
					# # print('aaa')

					d += 1
					# print(d == float('inf'))
					if d < float("inf"):
						if d < minDist:
							minDist = d
							e0 = ei
						u = self.vertexs[self.edges[ei].dst]
						while u.ts != curr_ts:
							# print(u.ts)
							u.ts = curr_ts
							d -= 1
							u.dist = d
							u = self.vertexs[self.edges[u.parent].dst]

					ei = self.edges[ei].next
					# print(ei)

				# print('aaabb')
				v2.parent = e0
				if v2.parent > 0:
					v2.ts = curr_ts
					v2.dist = minDist
					continue

				v2.ts = 0
				ei = v2.first
				while ei != 0:
					# print('a')
					u = self.vertexs[self.edges[ei].dst]
					ej = u.parent
					if u.t != vt or (not ej):
						ei = self.edges[ei].next
						continue
					if self.edges[ei^(vt^1)].weight and (not u.next):
						u.next = nilNode.get_value()
						# last = last.next = u
						last.get_value().next = u
						last.id = id(u)
					if ej > 0 and self.vertexs[self.edges[ej].dst] == v2:
						orphans.append(u)
						u.parent = ORPHAN
					ei = self.edges[ei].next
				# print(orphans)
		# # print([self.vertexs[i].t for i in range(len(self.vertexs))])
		# print([x.t for x in self.vertexs])
		return self.flow

	def insource_segment(self, i):
		return self.vertexs[i].t == 0

