#include <iostream>
#include <vector>
#include <string>
#include <cmath>



struct TreeNode ;
struct EdgeNode ;

typedef std::string tree_t ;

struct EdgeNode {
    tree_t val ;
    TreeNode* subtree ;
    EdgeNode* next ;
};

struct TreeNode {
    tree_t val ;
    EdgeNode* subtree_l ;
};



class A3Tree {

private:

    TreeNode* t ;
    std::vector<std::string> feature_names ; 
    int target_idx ;                         


    /*
      WHAT IT DOES: Finds most common classification label
      EXPLANATION: Used as a backup. If the tree runs out of features to split on 
      but the data still mixed, majority answer is default
     */
    std::string majority_outcome(const std::vector<std::vector<std::string>>& data) const {
        std::vector<std::string> vals ;
        std::vector<int> counts ;
        for (size_t i = 0; i < data.size(); ++i) {
            const std::string& v = data[i][target_idx] ;
            bool found = false ;
            for (size_t j = 0; j < vals.size() && !found; ++j) {
                if (vals[j] == v) { counts[j]++ ; found = true ; }}
            if (!found) { vals.push_back(v) ; counts.push_back(1) ; }}
        size_t best = 0 ;
        for (size_t j = 1; j < counts.size(); ++j) {
            if (counts[j] > counts[best]) best = j ;
        }
        return vals[best] ;
    }


    /*
      WHAT IT DOES: Measures data entropy 
      EXPLANATION: Returns 0.0 if all rows perfectly clean (pure)
      Returns a high value if outcomes are mixed. Used to calculate information gain
     */
    double entropy(const std::vector<std::vector<std::string>>& data) const {
        if (data.empty()) return 0.0 ;
        std::vector<std::string> vals ;
        std::vector<int> counts ;
        for (size_t i = 0; i < data.size(); ++i) {
            const std::string& v = data[i][target_idx] ;
            bool found = false ;
            for (size_t j = 0; j < vals.size() && !found; ++j) {
                if (vals[j] == v) { counts[j]++ ; found = true ; }}
            if (!found) { vals.push_back(v) ; counts.push_back(1) ; }}
        double e = 0.0 ;
        double n = (double)data.size() ;
        for (size_t j = 0; j < counts.size(); ++j) {
            double p = counts[j] / n ;
            e -= p * std::log2(p) ;
        }
        return e ;
    }


    /*
      WHAT IT DOES: Collects all unique category choices in a single column
      EXPLANATION: Tells the tree how many branches to build out of a node
     */
    std::vector<std::string> get_unique_vals(
            const std::vector<std::vector<std::string>>& data, int f) const {
        std::vector<std::string> vals ;
        for (size_t r = 0; r < data.size(); ++r) {
            const std::string& v = data[r][f] ;
            bool found = false ;
            for (size_t j = 0; j < vals.size() && !found; ++j) {
                if (vals[j] == v) found = true ;
            }
            if (!found) vals.push_back(v) ;
        }
        return vals ;
    }


    /*
      WHAT IT DOES: Chooses the best feature column to split the data on
      EXPLANATION: Loops through all remaining features, tracks maximum information gain
     */
    int best_feature(const std::vector<std::vector<std::string>>& data,
                     const std::vector<int>& features, double& max_gain) const {
        double base_e = entropy(data) ;
        max_gain = -1.0 ;
        int best_f = features[0] ;
        for (size_t i = 0; i < features.size(); ++i) {
            int f = features[i] ;
            std::vector<std::string> uv = get_unique_vals(data, f) ;
            double weighted_e = 0.0 ;
            for (size_t j = 0; j < uv.size(); ++j) {
                std::vector<std::vector<std::string>> subset ;
                for (size_t r = 0; r < data.size(); ++r) {
                    if (data[r][f] == uv[j]) subset.push_back(data[r]) ;
                }
                weighted_e += (double)subset.size() / (double)data.size() * entropy(subset) ;
            }
            double gain = base_e - weighted_e ;
            if (gain > max_gain) { max_gain = gain ; best_f = f ; }
        }
        return best_f ;
    }


    /*
      WHAT IT DOES: Deletes a node's outgoing branch edges
      EXPLANATION: Recursively deletes the attached subtrees first, then frees the linked list of choices
     */
    void delete_children(TreeNode* node) {
        EdgeNode* edge = node->subtree_l ;
        while (edge) {
            EdgeNode* nxt = edge->next ;
            delete_tree(edge->subtree) ;
            delete edge ;
            edge = nxt ;
        }}
    

    /*
      WHAT IT DOES: Recursively deletes the entire tree from the bottom up
      EXPLANATION: Frees memory for children before deleting the parent node, prevent memory leaks 
     */
    void delete_tree(TreeNode* node) {
        if (!node) return ;
        delete_children(node) ;
        delete node ;
    }


    /*
      WHAT IT DOES: Builds decision tree
      EXPLANATION: Uses entropy for purity metrics and halts splitting if info gain drops to 0
     */
    TreeNode* build(const std::vector<std::vector<std::string>>& data,
                    std::vector<int> available) {
        TreeNode* node  = new TreeNode ;
        node->subtree_l = NULL ;

        if (data.empty())          { node->val = "" ;                    return node ; }
        if (entropy(data) == 0.0)  { node->val = data[0][target_idx] ;   return node ; }
        if (available.empty())     { node->val = majority_outcome(data) ; return node ; }

        double max_gain = 0.0;
        int f = best_feature(data, available, max_gain) ;
        

        // Pre-pruning: If no predictive progress can be made, collapse to a leaf immediately
        if (max_gain <= 0.0) {
            node->val = majority_outcome(data);
            return node;
        }

        node->val = feature_names[f] ;
        std::vector<std::string> uv = get_unique_vals(data, f) ;

        std::vector<int> remaining ;
        for (size_t i = 0; i < available.size(); ++i) {
            if (available[i] != f) remaining.push_back(available[i]) ;
        }

        for (size_t j = 0; j < uv.size(); ++j) {
            std::vector<std::vector<std::string>> subset ;
            for (size_t r = 0; r < data.size(); ++r) {
                if (data[r][f] == uv[j]) subset.push_back(data[r]) ;
            }

            EdgeNode* edge  = new EdgeNode ;
            edge->val       = uv[j] ;
            edge->subtree   = NULL ;          
            edge->next      = node->subtree_l ;
            node->subtree_l = edge ;
            edge->subtree   = build(subset, remaining) ;
        }
        return node ;
    }


    /*
      WHAT IT DOES: Recursively counts every node in the tree
      EXPLANATION: Counts all decision questions and all leaves combined
     */
    int count_nodes(TreeNode* node) const {
        if (!node) return 0 ;
        int c = 1 ;
        EdgeNode* edge = node->subtree_l ;
        while (edge) { c += count_nodes(edge->subtree) ; edge = edge->next ; }
        return c ;
    }


    /*
      WHAT IT DOES: Recursively counts only the leaves
      EXPLANATION: Count nodes that have zero outgoing branches
     */
    int count_leaves(TreeNode* node) const {
        if (!node) return 0 ;
        if (!node->subtree_l) return 1 ;
        int c = 0 ;
        EdgeNode* edge = node->subtree_l ;
        while (edge) { c += count_leaves(edge->subtree) ; edge = edge->next ; }
        return c ;
    }


    /*
      WHAT IT DOES: Prints the tree to the console
      EXPLANATION: Uses indented spaces to show how questions branch down into final answers
     */
    void print_tree(TreeNode* node, int depth) const {
        if (!node) return ;
        if (!node->subtree_l) {
            std::cout << node->val << std::endl ;
            return ;
        }
        std::cout << node->val << std::endl ;
        EdgeNode* edge = node->subtree_l ;
        while (edge) {
            for (int i = 0 ; i < (depth + 1) * 2 ; ++i) {
                std::cout << " " ;
            }
            std::cout << edge->val << " -> " ;
            if (edge->subtree && !edge->subtree->subtree_l) {
                std::cout << edge->subtree->val << std::endl ;
            } else if (edge->subtree) {
                print_tree(edge->subtree, depth + 1) ;
            }
            edge = edge->next ;
        }}



public:


    /*
      WHAT IT DOES: Constructor
      EXPLANATION: Reads column headers, sets up available feature indexes, and builds tree
     */
    A3Tree(const std::vector<std::vector<std::string>>& input) : t(NULL) {
        target_idx = input[0].size() - 1 ;
        for (int i = 0; i < target_idx; ++i) {
            feature_names.push_back(input[0][i]) ;
        }

        std::vector<std::vector<std::string>> data(input.begin() + 1, input.end()) ;

        std::vector<int> available ;
        for (int i = 0; i < target_idx; ++i) available.push_back(i) ;

        t = build(data, available) ;
    }


    /*
      WHAT IT DOES: Destructor 
      EXPLANATION: Deletes all nodes and branches when the tree object is destroyed
     */
    ~A3Tree() { delete_tree(t) ; }


    /*
      WHAT IT DOES: Predicts the answer for a new sample row
      EXPLANATION: Starts at the root, looks up the sample's value for the current 
      feature question, follows the matching branch edge down, and repeats until it hits a leaf
     */
    std::string query(const std::vector<std::string>& q) const {
        TreeNode* cur = t ;
        while (cur && cur->subtree_l) {
            int f_idx = -1 ;
            for (size_t i = 0; i < feature_names.size() && f_idx == -1; ++i) {
                if (feature_names[i] == cur->val) f_idx = (int)i ;
            }
            if (f_idx == -1) return "" ;

            EdgeNode* edge = cur->subtree_l ;
            TreeNode* nxt  = NULL ;
            while (edge && !nxt) {
                if (edge->val == q[f_idx]) nxt = edge->subtree ;
                edge = edge->next ;
            }
            cur = nxt ;
        }
        return cur ? cur->val : "" ;
    }


    int node_count()      const { return count_nodes(t) ; }
    int leaf_node_count() const { return count_leaves(t) ; }
    void print()          const { print_tree(t, 0) ; }
};



int main() {

    std::vector<std::vector<std::string>> input1 = {
        {"temperature", "rain", "wind", "quality"},
        {"high", "yes", "light",    "acceptable"},
        {"low",  "yes", "light",    "acceptable"},
        {"low",  "no",  "moderate", "good"},
        {"high", "yes", "strong",   "poor"},
        {"high", "yes", "moderate", "acceptable"},
        {"high", "no",  "moderate", "good"},
        {"low",  "yes", "strong",   "poor"},
        {"high", "no",  "light",    "good"},
        {"low",  "yes", "moderate", "poor"},
        {"high", "no",  "strong",   "poor"}
    } ;

    A3Tree t1(input1) ;
    std::cout << "=== Test 1: kayaking dataset ===" << std::endl ;
    std::cout << t1.query({"low", "yes", "strong"}) << std::endl ;   // expect: poor
    std::cout << t1.query({"high", "yes", "moderate"}) << std::endl ; // expect: acceptable
    std::cout << t1.query({"low", "no", "moderate"}) << std::endl ;   // expect: good
    std::cout << t1.query({"high", "no", "light"}) << std::endl ;     // expect: good
    std::cout << t1.node_count()      << std::endl ; // expect: 10
    std::cout << t1.leaf_node_count() << std::endl ; // expect: 6
    std::cout << "\n--- Visual Tree Layout ---" << std::endl;
    t1.print();



    std::vector<std::vector<std::string>> input2 = {
        {"Feature_3", "feature2", "feature", "feature0", "not_a_feature"},
        {"a13482",  "10",      "a13480",   "a", "1"},
        {"B_34201", "9",       "1343435",  "a", "a2"},
        {"a13482",  "8",       "57357",    "a", "3"},
    } ;

    A3Tree t2(input2) ;
    std::cout << "\n=== Test 2: unusual values ===" << std::endl ;
    std::cout << t2.query({"B_34201", "9", "1343435", "a"}) << std::endl ; // expect: a2
    std::cout << t2.query({"a13482",  "8", "57357",   "a"}) << std::endl ; // expect: 3
    t2.print();



    std::vector<std::vector<std::string>> input3 = {
        {"color", "label"},
        {"red",   "A"},
        {"blue",  "B"},
        {"red",   "A"}
    } ;

    A3Tree t3(input3) ;
    std::cout << "\n=== Test 3: single-feature dataset ===" << std::endl ;
    std::cout << t3.query({"red"})  << std::endl ; // expect: A
    std::cout << t3.query({"blue"}) << std::endl ; // expect: B
    std::cout << t3.query({"orange"}) << std::endl ; // expect: ""
    std::cout << t3.node_count()    << std::endl ; // expect: 3



    std::vector<std::vector<std::string>> input4 = {
        {"x", "y", "result"},
        {"a", "p", "yes"},
        {"b", "p", "yes"},
        {"a", "q", "yes"}
    } ;

    A3Tree t4(input4) ;
    std::cout << "\n=== Test 4: uniform outcome (pruning should yield 1 node) ===" << std::endl ;
    std::cout << t4.query({"a", "p"}) << std::endl ; // expect: yes
    std::cout << t4.node_count()      << std::endl ; // expect: 1
    std::cout << t4.leaf_node_count() << std::endl ; // expect: 1



    std::vector<std::vector<std::string>> input5 = {
        {"constant_feature", "outcome"},
        {"same_value", "yes"},
        {"same_value", "yes"},
        {"same_value", "no"},
        {"same_value", "no"},
        {"same_value", "no"}
    } ;

    A3Tree t5(input5) ;
    std::cout << "\n=== Test 5: exhausted features with mixed labels (majority default) ===" << std::endl ;
    std::cout << t5.query({"same_value"}) << std::endl ; // expect: no
    std::cout << t5.node_count()          << std::endl ; // expect: 1 
    std::cout << t5.leaf_node_count()     << std::endl ; // expect: 1

    return 0;
}