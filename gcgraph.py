# GCGRAPH

class Vertex:
	def __init__(self):
		self.next = None # Initialized and used in maxflow() only
		self.parent = 0
		self.first = 0
		self.ts = 0
		self.dist = 0
		self.weight = 0.0
		self.t = None

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
		sefl.vertex_count = vertex_count
		self.edge_count = edge_count

	def add_vertex(self):
		v = Vertex()
		self.vertexs.append(v)

	def add_edges(self, i, j, w, revw):

		if len(self.edges == 0):
			self.edges = [0, 0]

		fromI = Edge()
		fromI.dst = j
		fromI.next = self.vertexs[i].first
		fromI.weight = w
		vertexs[i].first = len(self.edges)
		self.edges.append(fromI)

		toI = Edge()
		toI.dst = i
		toI.next = self.vertexs[j].first
		toI.weight = revw
		self.vertexs[j].first = len(self.edges)
		self.edges.append(toI)

	def add_term_weights(self, i, source_weight, sink_weight):
		