The goal is to build state transition graphs, going from an initial state to a
final state

A state transition graph consists of nodes and edges/lines 
(in correspondence to feynman graphs): 
- The connection lines we call particle lines, which are basically a list of
 quantum numbers (QN) that define the particle state (That list can be empty
 at first).
- The nodes are of type InteractionNode, that contain all information for the
 transition of this specific step. An interaction node has M ingoing lines and
 N outgoing lines (M,N = Integer & M > 0, N > 0) 

##Concept of building graphs:

###Step 1:
Building of all possible topologies. A topology is a graph, in which the edges
and nodes are empty (no QN information). See the topology sub-modules.

###Step 2:
Filling the toplogy graphs with QN information. This means initializing the 
topology graphs with the initial and final state quantum numbers and
propagating these through the complete graph. Here also the combinatorics of
the initial and final state should be taken into account.

###Step 3:
Duplicate the graphs and insert concrete particles for the edges (inserting
the mass variable)

###Step 4:
Output to XML model file
