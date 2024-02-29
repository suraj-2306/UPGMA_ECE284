class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child):
        if isinstance(child, TreeNode):
            self.children.append(child)
        else:
            print("Child must be a TreeNode")

    def print_tree(self, level=0):
        print('  ' * level + str(self.data))
        for child in self.children:
            child.print_tree(level + 1)

def build_cluster_tree(progression):
    root = TreeNode("Root")  # Create a root node for the tree
    current_nodes = [TreeNode(data) for data in progression[0]]  # Create nodes for initial clusters

    for clusters in progression[1:]:
        new_nodes = []  # List to store new nodes for merged clusters
        i = 0
        while i < len(clusters):
            merged_cluster_data = clusters[i] + (clusters[i+1] if i+1 < len(clusters) else [])
            merged_cluster_node = TreeNode(merged_cluster_data)
            new_nodes.append(merged_cluster_node)
            if i+1 < len(clusters):
                merged_cluster_node.add_child(current_nodes[i])
                merged_cluster_node.add_child(current_nodes[i+1])
            else:
                merged_cluster_node.add_child(current_nodes[i])
            i += 2

        current_nodes = new_nodes  # Update current nodes for the next iteration

    root.children = current_nodes  # Set the merged clusters as children of the root
    return root

# Example usage:
progression = [[['a'], ['b'], ['c'], ['d']],
               [['a', 'b'], ['c'], ['d']],
               [['a', 'b'], ['c', 'd']],
               [['a', 'b', 'c', 'd']]]

# Build the cluster tree
cluster_tree = build_cluster_tree(progression)

# Print the cluster tree
print("Cluster tree:")
cluster_tree.print_tree()
