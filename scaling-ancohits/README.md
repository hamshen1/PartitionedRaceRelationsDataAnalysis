# README

Paper: [Partisan Scale: Discovering Partisanship in the U.S. Senate Using Signed Bipartite Graphs](https://dl.acm.org/doi/pdf/10.1145/2187980.2188046)

**ANCOHITS.m:** MATLAB implementation of the ANCO-HITS algorithm for co-scaling signed bipartite graphs.  
- **Input:** Matrix `A` (signed bipartite graph), tolerance `epsilon`.  
- **Output:** Score vectors `s1`, `s2`.  

**Anco_HIT_Algorithm.py:** Python implementation of the ANCO-HITS algorithm.  
- **Input:** Pickled matrix file (signed bipartite graph), tolerance `epsilon`.  
- **Output:** Score vectors `s1`, `s2` saved as `.pkl` and `.txt` files.  
