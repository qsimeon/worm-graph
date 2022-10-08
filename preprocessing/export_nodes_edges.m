%% Saves as separate .csv tables the nodes and edges of chemical synsapse and gap junctins in the C.elegans hermaphrodite
load("GHermChem.mat") % chemical synapses (directed graph)
load("GHermElec_Sym.mat") % gap junctions (indirected graph)

% Write as .csv in order to load as a Pandas dataframe in Python
writetable(GHermChem.Edges, 'GHermChem_Edges.csv')
writetable(GHermChem.Nodes, 'GHermChem_Nodes.csv')

writetable(GHermElec_Sym.Nodes, 'GHermElec_Sym_Nodes.csv')
writetable(GHermElec_Sym.Edges, 'GHermElec_Sym_Edges.csv')