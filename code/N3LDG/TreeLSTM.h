#ifndef TREELSTM
#define TREELSTM

#include <set>
#include "MyLib.h"
#include "Node.h"
#include "BiOP.h"
#include "AtomicOP.h"
#include "Graph.h"
#include "PMultiOP.h"
#include "PAddOP.h"
#include "BucketOP.h"
#include "./util/Util-options.h"

using namespace egstra;


class TreeNode {
public:
	/*int id;*/
	int degree;
	int parent;
	vector<int> children;
};

inline int find_least_common_node(const vector<TreeNode> &tree, int i, int j) {
	vector<int> iparents;
	vector<int> jparents;

	iparents.push_back(i);
	while (i != -1) {
		iparents.push_back(tree[i].parent);
		i = tree[i].parent;
	}

	jparents.push_back(j);
	while (j != -1) {
		jparents.push_back(tree[j].parent);
		j = tree[j].parent;
	}

	for (int m = 0; m < iparents.size(); m++) {
		for (int n = 0; n < jparents.size(); n++) {
			if (iparents[m] == jparents[n]) {
				return iparents[m];
			}
		}
	}
}

inline void find_sp(const vector<TreeNode> &tree, int i, int lcn, vector<int>& path) {  // (i --> lcn)
	path.push_back(i);
	
	while (i != lcn) {
		path.push_back(tree[i].parent);
		i = tree[i].parent;
	}
}
//class Tree {
//public:
//	vector<TreeNode> elems;
//};

struct TreeLSTMParams{
	BiParams input;
	BiParams output;
	BiParams forget;
	BiParams cell;
	LookupTable left_init_state_params;
	LookupTable right_init_state_params;
	LookupTable lcn_init_state_params;
	
	int flag;  //  0,  origin-init-state ; 1 , share-init-state; 2, buckets

	TreeLSTMParams() {
		assert(options::get("flag", flag));
	}

	inline void exportAdaParams(ModelUpdate& ada, bool bu = true) {
		input.exportAdaParams(ada);
		output.exportAdaParams(ada);
		forget.exportAdaParams(ada);
		cell.exportAdaParams(ada);
		if (flag == 0 || flag == 1) {  // 只有 0, 1的 情况需要更新参数
			if (bu) {
				left_init_state_params.exportAdaParams(ada);
				right_init_state_params.exportAdaParams(ada);
			}
			else {
				lcn_init_state_params.exportAdaParams(ada);
			}
		}
	}

	inline void initial(int nOSize, int nISize, bool bu = true) {
		input.initial(nOSize, nOSize, nISize, true);
		output.initial(nOSize, nOSize, nISize, true);
		forget.initial(nOSize, nOSize, nISize, true);
		cell.initial(nOSize, nOSize, nISize, true);
		left_init_state_params.initial_constant(2, nOSize, bu);
		right_init_state_params.initial_constant(2, nOSize, bu);
		lcn_init_state_params.initial_constant(2, nOSize, !bu);
		
	}

	inline int inDim() {
		return input.W2.inDim();
	}

	inline int outDim() {
		return input.W2.outDim();
	}

	inline void save(std::ofstream &os) const {
		input.save(os);
		output.save(os);
		forget.save(os);
		cell.save(os);
		left_init_state_params.save(os);
		right_init_state_params.save(os);
		lcn_init_state_params.save(os);
	}

	inline void load(std::ifstream &is) {
		input.load(is);
		output.load(is);
		forget.load(is);
		cell.load(is);
		left_init_state_params.load(is);
		right_init_state_params.load(is);
		lcn_init_state_params.load(is);
	}
};

class TreeLSTMBuilder {
public:
	dtype _dropout;
	int _nSize;
	int _inDim;
	int _outDim;

	vector<PAddNode> _input_children_addition;
	vector<BiNode> _inputgates; // 
	vector<vector<BiNode>> _forgetgates;
	vector<BiNode> _halfcells;  //len

	vector<PMultiNode> _inputfilters; //len
	vector<vector<PMultiNode>> _forgetfilters;
	vector<PAddNode> _forgetfilters_addition;

	vector<PAddNode> _cells;
	vector<BiNode> _outputgates;
	vector<TanhNode> _halfhiddens;
	vector<PMultiNode> _hiddens;  // intermediate result without dropout

	BucketNode _bucket;
	LookupNode left_hidden_init;
	LookupNode left_cell_init;
	LookupNode right_hidden_init;
	LookupNode right_cell_init;
	LookupNode lcn_hidden_init;
	LookupNode lcn_cell_init;
	
	TreeLSTMParams* _param;

	bool _bottom_up;

	int flag;

public:
	TreeLSTMBuilder() {
		assert(options::get("flag", flag));
		clear();
	}

	~TreeLSTMBuilder() {
		clear();
	}


	inline void resize(int maxsize) {
		_input_children_addition.resize(maxsize);
		_inputgates.resize(maxsize);
		_forgetgates.resize(maxsize);
		for (int i = 0; i < maxsize; i++) {
			_forgetgates[i].resize(2);
		}
		_halfcells.resize(maxsize);
		_inputfilters.resize(maxsize);
		_forgetfilters.resize(maxsize);
		for (int i = 0; i < maxsize; i++) {
			_forgetfilters[i].resize(2);
		}
		_forgetfilters_addition.resize(maxsize);
		_cells.resize(maxsize);
		_outputgates.resize(maxsize);
		_halfhiddens.resize(maxsize);
		_hiddens.resize(maxsize);
	}

	inline void clear() {
		_input_children_addition.clear();
		_inputgates.clear();
		_forgetgates.clear();
		_halfcells.clear();

		_inputfilters.clear();
		_forgetfilters.clear();
		_forgetfilters_addition.clear();

		_cells.clear();
		_outputgates.clear();
		_halfhiddens.clear();
		_hiddens.clear();  // intermediate result without dropout

		_bottom_up = true;
		_param = NULL;
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
	}

	inline void init(TreeLSTMParams* paramInit, dtype dropout, bool bottom_up = true) {
		_param = paramInit;
		_inDim = _param->input.W2.inDim();
		_outDim = _param->input.W2.outDim();
		int maxsize = _inputgates.size();

		if (bottom_up) {
			left_hidden_init.setParam(&(paramInit->left_init_state_params));
			left_cell_init.setParam(&(paramInit->left_init_state_params));
		
			if (flag == 0) { //i, j 的init-state 不一样
				right_hidden_init.setParam(&(paramInit->right_init_state_params));
				right_cell_init.setParam(&(paramInit->right_init_state_params));
			}
			else if (flag == 1 || flag == 2) { //i, j 的init-state 一样 或者 不更新（就是buckets）
				right_hidden_init.setParam(&(paramInit->left_init_state_params));
				right_cell_init.setParam(&(paramInit->left_init_state_params));
			}
		}
		else {
			lcn_hidden_init.setParam(&(paramInit->lcn_init_state_params));
			lcn_cell_init.setParam(&(paramInit->lcn_init_state_params));
		}

		for (int idx = 0; idx < maxsize; idx++) {
			_inputgates[idx].setParam(&_param->input);
			for (int idy = 0; idy < 2; idy++) {
				_forgetgates[idx][idy].setParam(&_param->forget);
			}
			_outputgates[idx].setParam(&_param->output);
			_halfcells[idx].setParam(&_param->cell);

			_inputgates[idx].setFunctions(&fsigmoid, &dsigmoid);
			for (int idy = 0; idy < 2; idy++) {
				_forgetgates[idx][idy].setFunctions(&fsigmoid, &dsigmoid);
			}
			_outputgates[idx].setFunctions(&fsigmoid, &dsigmoid);
			_halfcells[idx].setFunctions(&ftanh, &dtanh);
		}
		_bottom_up = bottom_up;

		//for (int idx = 0; idx < maxsize; idx++) {
		//	_input_children_addition[idx].init(_outDim, -1);
		//	_inputgates[idx].init(_outDim, -1);
		//	/*for (int idy = 0; idy < _forgetgates[idx].size(); idy++) {
		//		_forgetgates[idx][idy].init(_outDim, -1);
		//	}*/
		//	_halfcells[idx].init(_outDim, -1);
		//	_inputfilters[idx].init(_outDim, -1);
		//	/*for (int idy = 0; idy < _forgetfilters[idx].size(); idy++) {
		//		_forgetfilters[idx][idy].init(_outDim, -1);
		//	}*/
		//	_forgetfilters_addition[idx].init(_outDim, -1);
		//	_cells[idx].init(_outDim, -1);
		//	_outputgates[idx].init(_outDim, -1);
		//	_halfhiddens[idx].init(_outDim, -1);
		//	_hiddens[idx].init(_outDim, dropout);
		//}


		_bucket.init(_outDim, -1);
		
		_dropout = dropout;
	}

	void malloc_nodes(int index, bool isTree) {
		if(isTree) _input_children_addition[index].init(_outDim, -1);
		_inputgates[index].init(_outDim, -1);
		if (!isTree) {
			_forgetgates[index][0].init(_outDim, -1);
		}
		else {
			_forgetgates[index][0].init(_outDim, -1);
			_forgetgates[index][1].init(_outDim, -1);
		}
		_halfcells[index].init(_outDim, -1);
		_inputfilters[index].init(_outDim, -1);

		if (!isTree) {
			_forgetfilters[index][0].init(_outDim, -1);
		}
		else {
			_forgetfilters[index][0].init(_outDim, -1);
			_forgetfilters[index][1].init(_outDim, -1);
		}

		
		if(isTree)  _forgetfilters_addition[index].init(_outDim, -1);

		_cells[index].init(_outDim, -1);
		_outputgates[index].init(_outDim, -1);
		_halfhiddens[index].init(_outDim, -1);
		_hiddens[index].init(_outDim, _dropout); // !!! dropout -1
	}



	void malloc_path_nodes(int i, int j, const vector<TreeNode>& tree, bool bt) {
		if (bt) {
			left_hidden_init.init(_outDim, _dropout);
			left_cell_init.init(_outDim, _dropout);

			right_hidden_init.init(_outDim, _dropout);
			right_cell_init.init(_outDim, _dropout);
		}
		else {
			lcn_hidden_init.init(_outDim, _dropout);
			lcn_cell_init.init(_outDim, _dropout);
		}


		if (i == j) {
			malloc_nodes(i, false);
			return;
		}

		int lcn = find_least_common_node(tree, i, j);

		if (lcn == i) {
			vector<int> path;
			find_sp(tree, j, lcn, path);
			for (int idx = 0; idx < path.size(); idx++) {
				malloc_nodes(path[idx], false);
			}
			return;
		}

		if (lcn == j) {
			vector<int> path;
			find_sp(tree, i, lcn, path);
			for (int idx = 0; idx < path.size(); idx++) {
				malloc_nodes(path[idx], false);
			}
			return ;
		}

		vector<int> left_path;
		find_sp(tree, i, lcn, left_path);
		for (int idx = 0; idx < left_path.size() - 1; idx++) {
			malloc_nodes(left_path[idx], false);
		}

		vector<int> right_path;
		find_sp(tree, j, lcn, right_path);
		for (int idx = 0; idx < right_path.size() - 1; idx++) {
			malloc_nodes(right_path[idx], false);
		}

		malloc_nodes(lcn, bt);
	}


	//whether vectors have been allocated
	inline bool empty() {
		return _hiddens.empty();
	}
public:
	inline void forward(Graph *cg, const vector<PNode>&x,const vector<TreeNode>& tree) {
		if (x.size() == 0) {
			std::cout << "empty inputs for lstm operation" << std::endl;
			return;
		}
		_nSize = x.size();

		if (x[0]->val.dim != _inDim) {
			std::cout << "input dim does not match for lstm operation" << std::endl;
			return;
		}

		if (_bottom_up) {
			bottom_up_forward(cg, x, tree);
		}
		else {
			top_down_forward(cg, x, tree);
		}
	}

	inline void forward(Graph *cg, const vector<PNode>&x, const vector<TreeNode>& tree, int i, int j, int& lcn) {
		if (x.size() == 0) {
			std::cout << "empty inputs for lstm operation" << std::endl;
			return;
		}
		_nSize = x.size();

		if (x[0]->val.dim != _inDim) {
			cout << x[0]->val.dim << _inDim << endl;
			std::cout << "input dim does not match for lstm operation" << std::endl;
			return;
		}

		if (_bottom_up) {
			bottom_up_forward(cg, x, tree, i, j, lcn);
		}
		else {
			top_down_forward(cg, x, tree, i, j);
		}
	}

protected:
	inline void bottom_up_forward(Graph *cg, const vector<PNode>& x, const vector<TreeNode>& _tree_) {
		vector<TreeNode> tree = _tree_;

	    vector<int> current;
		

		for (int i = 0; i < tree.size(); i++) {
			if (tree[i].degree == 0) {
				/*assert(i == tree[i].id);*/
				current.push_back(i);
			}
		}

		for (int i = 0; i < current.size(); i++) {
			int t = current[i];
			
			_bucket.forward(cg, 0);
			_inputgates[t].forward(cg, &_bucket, x[t]);
			_halfcells[t].forward(cg, &_bucket, x[t]);
			_inputfilters[t].forward(cg, &_halfcells[t], &_inputgates[t]);
			_cells[t].forward(cg, &_inputfilters[t], &_bucket);
			_halfhiddens[t].forward(cg, &_cells[t]);
			_outputgates[t].forward(cg, &_bucket, x[t]);
			_hiddens[t].forward(cg, &_halfhiddens[t], &_outputgates[t]);


			tree[tree[t].parent].degree--;
			tree[t].degree = -1;
		}

		current.clear();
		for (int i = 0; i < tree.size(); i++) {
			if (tree[i].degree == 0) {
				current.push_back(i);
			}
		}


		while (!current.empty()) {
			for (int i = 0; i < current.size(); i++) {
				int t = current[i];
				vector<int> children = tree[t].children;
				vector<PNode> children_pnode;
				for (int j = 0; j < children.size(); j++) {
					children_pnode.push_back(&_hiddens[children[j]]);
				}
				_input_children_addition[t].forward(cg, children_pnode);
				_inputgates[t].forward(cg, &_input_children_addition[t], x[t]);
				_outputgates[t].forward(cg, &_input_children_addition[t], x[t]);
				_halfcells[t].forward(cg, &_input_children_addition[t], x[t]);


				_forgetgates[t].clear();
				_forgetgates[t].resize(children.size());

				for (int j = 0; j < children.size(); j++) {
					_forgetgates[t][j].setParam(&_param->forget);
					_forgetgates[t][j].setFunctions(&fsigmoid, &dsigmoid);
					_forgetgates[t][j].init(_outDim, -1);
					_forgetgates[t][j].forward(cg, &_hiddens[children[j]], x[t]);
				}

				_inputfilters[t].forward(cg, &_halfcells[t], &_inputgates[t]);


				_forgetfilters[t].clear();
				_forgetfilters[t].resize(children.size());
				for (int j = 0; j < children.size(); j++) {
					_forgetfilters[t][j].init(_outDim, -1);
					_forgetfilters[t][j].forward(cg,  &_cells[children[j]], &_forgetgates[t][j]);
				}
				
				_forgetfilters_addition[t].forward(cg, getPNodes(_forgetfilters[t], children.size()));

				_cells[t].forward(cg, &_inputfilters[t], &_forgetfilters_addition[t]);

				_halfhiddens[t].forward(cg, &_cells[t]);

				_hiddens[t].forward(cg, &_halfhiddens[t], &_outputgates[t]);

				if (tree[t].parent >= 0) {
					tree[tree[t].parent].degree--;
				}
				tree[t].degree = -1;

			}

			current.clear();
			for (int i = 0; i < tree.size(); i++) {
				if (tree[i].degree == 0) {
					current.push_back(i);
				}
			}
		}
	}

	inline void bottom_up_forward(Graph *cg, const vector<PNode>& x, const vector<TreeNode>& _tree_, int i, int j, int &lcn) {
		vector<TreeNode> tree = _tree_;
		malloc_path_nodes(i, j, tree, true);
		if (i == j) {
			_bucket.forward(cg, 0);
			_inputgates[i].forward(cg, &_bucket, x[i]);
			_halfcells[i].forward(cg, &_bucket, x[i]);
			_inputfilters[i].forward(cg, &_halfcells[i], &_inputgates[i]);
			_cells[i].forward(cg, &_inputfilters[i], &_bucket);
			_halfhiddens[i].forward(cg, &_cells[i]);
			_outputgates[i].forward(cg, &_bucket, x[i]);
			_hiddens[i].forward(cg, &_halfhiddens[i], &_outputgates[i]);
		
			lcn = i;	
			return;
		}

		/*if (lcn != i) {
			left_input_init.forward(cg, 0);
			left_forget_init.forward(cg, 1);
			left_output_init.forward(cg, 2);
			left_halfcells_init.forward(cg, 3);
			left_cells_init.forward(cg, 4);
			
		}
		if (lcn != j) {
			right_input_init.forward(cg, 0);
			right_forget_init.forward(cg, 1);
			right_output_init.forward(cg, 2);
			right_halfcells_init.forward(cg, 3);
			right_cells_init.forward(cg, 4);
		}*/

		lcn = find_least_common_node(tree, i, j);

		vector<int> current;

		int left = i;
		int pre;

		while (left != lcn) {
			if (left == i) {
				left_cell_init.forward(cg, 0);
				left_hidden_init.forward(cg, 1);
				
				_inputgates[left].forward(cg, &left_hidden_init, x[left]);
				_forgetgates[left][0].forward(cg, &left_hidden_init, x[left]);
				_halfcells[left].forward(cg, &left_hidden_init, x[left]);
				_outputgates[left].forward(cg, &left_hidden_init, x[left]);

				_inputfilters[left].forward(cg, &_halfcells[left], &_inputgates[left]);
				_forgetfilters[left][0].forward(cg, &left_cell_init, &_forgetgates[left][0]);
				_cells[left].forward(cg, &_inputfilters[left], &_forgetfilters[left][0]);
				_halfhiddens[left].forward(cg, &_cells[left]);
				_hiddens[left].forward(cg, &_halfhiddens[left], &_outputgates[left]);
			}
			else {
				_inputgates[left].forward(cg, &_hiddens[pre], x[left]); 

		/*		_forgetgates[left].clear();
				_forgetgates[left].resize(1);
				_forgetgates[left][0].setParam(&_param->forget);
				_forgetgates[left][0].setFunctions(&fsigmoid, &dsigmoid);
				_forgetgates[left][0].init(_outDim, -1);*/
				_forgetgates[left][0].forward(cg, &_hiddens[pre], x[left]);


				_halfcells[left].forward(cg, &_hiddens[pre], x[left]);
				_inputfilters[left].forward(cg, &_halfcells[left], &_inputgates[left]);

		/*		_forgetfilters[left].clear();
				_forgetfilters[left].resize(1);
				_forgetfilters[left][0].init(_outDim, -1);*/
				_forgetfilters[left][0].forward(cg, &_cells[pre], &_forgetgates[left][0]);

				_cells[left].forward(cg, &_inputfilters[left], &_forgetfilters[left][0]);

				_halfhiddens[left].forward(cg, &_cells[left]);
				_outputgates[left].forward(cg, &_hiddens[pre], x[left]);
				_hiddens[left].forward(cg, &_halfhiddens[left], &_outputgates[left]);
			}
			if (tree[left].parent == lcn) {
				break;
			}

			pre = left;
			left = tree[left].parent;
		}

		
		int right = j;
		while (right != lcn) {
			if (right == j) {
				right_cell_init.forward(cg, 0);
				right_hidden_init.forward(cg, 1);
		
				_inputgates[right].forward(cg, &right_hidden_init, x[right]);
				_forgetgates[right][0].forward(cg, &right_hidden_init, x[right]);
				_outputgates[right].forward(cg, &right_hidden_init, x[right]);
				_halfcells[right].forward(cg, &right_hidden_init, x[right]);

				_inputfilters[right].forward(cg, &_halfcells[right], &_inputgates[right]);
				_forgetfilters[right][0].forward(cg, &right_cell_init, &_forgetgates[right][0]);
				_cells[right].forward(cg, &_inputfilters[right], &_forgetfilters[right][0]);
				_halfhiddens[right].forward(cg, &_cells[right]);
				_hiddens[right].forward(cg, &_halfhiddens[right], &_outputgates[right]);


				/*_bucket.forward(cg, 0);
				_inputgates[right].forward(cg, &_bucket, x[right]);
				_halfcells[right].forward(cg, &_bucket, x[right]);
				_inputfilters[right].forward(cg, &_halfcells[right], &_inputgates[right]);
				_cells[right].forward(cg, &_inputfilters[right], &_bucket);
				_halfhiddens[right].forward(cg, &_cells[right]);
				_outputgates[right].forward(cg, &_bucket, x[right]);
				_hiddens[right].forward(cg, &_halfhiddens[right], &_outputgates[right]);*/
			}
			else {
				_inputgates[right].forward(cg, &_hiddens[pre], x[right]);

				/*_forgetgates[right].clear();
				_forgetgates[right].resize(1);
				_forgetgates[right][0].setParam(&_param->forget);
				_forgetgates[right][0].setFunctions(&fsigmoid, &dsigmoid);
				_forgetgates[right][0].init(_outDim, -1);*/
				_forgetgates[right][0].forward(cg, &_hiddens[pre], x[right]);

				_halfcells[right].forward(cg, &_hiddens[pre], x[right]);
				_inputfilters[right].forward(cg, &_halfcells[right], &_inputgates[right]);

			/*	_forgetfilters[right].clear();
				_forgetfilters[right].resize(1);
				_forgetfilters[right][0].init(_outDim, -1);*/
				_forgetfilters[right][0].forward(cg, &_cells[pre], &_forgetgates[right][0]);

				_cells[right].forward(cg, &_inputfilters[right], &_forgetfilters[right][0]);

				_halfhiddens[right].forward(cg, &_cells[right]);
				_outputgates[right].forward(cg, &_hiddens[pre], x[right]);
				_hiddens[right].forward(cg, &_halfhiddens[right], &_outputgates[right]);
			}
			if (tree[right].parent == lcn) {
				break;
			}

			pre = right;
			right = tree[right].parent;
		}

		if (lcn == i) {
			assert(i != j);
			_inputgates[lcn].forward(cg, &_hiddens[right], x[lcn]);

		/*	_forgetgates[lcn].clear();
			_forgetgates[lcn].resize(1);
			_forgetgates[lcn][0].setParam(&_param->forget);
			_forgetgates[lcn][0].setFunctions(&fsigmoid, &dsigmoid);
			_forgetgates[lcn][0].init(_outDim, -1);*/
			_forgetgates[lcn][0].forward(cg, &_hiddens[right], x[lcn]);

			_halfcells[lcn].forward(cg, &_hiddens[right], x[lcn]);
			_inputfilters[lcn].forward(cg, &_halfcells[lcn], &_inputgates[lcn]);

			/*_forgetfilters[lcn].clear();
			_forgetfilters[lcn].resize(1);
			_forgetfilters[lcn][0].init(_outDim, -1);*/
			_forgetfilters[lcn][0].forward(cg, &_cells[right], &_forgetgates[lcn][0]);

			_cells[lcn].forward(cg, &_inputfilters[lcn], &_forgetfilters[lcn][0]);

			_halfhiddens[lcn].forward(cg, &_cells[lcn]);
			_outputgates[lcn].forward(cg, &_hiddens[right], x[lcn]);
			_hiddens[lcn].forward(cg, &_halfhiddens[lcn], &_outputgates[lcn]);
		}
		else if (lcn == j) {
			assert(j != i);
			_inputgates[lcn].forward(cg, &_hiddens[left], x[lcn]);

			/*_forgetgates[lcn].clear();
			_forgetgates[lcn].resize(1);
			_forgetgates[lcn][0].setParam(&_param->forget);
			_forgetgates[lcn][0].setFunctions(&fsigmoid, &dsigmoid);
			_forgetgates[lcn][0].init(_outDim, -1);*/
			_forgetgates[lcn][0].forward(cg, &_hiddens[left], x[lcn]);

			_halfcells[lcn].forward(cg, &_hiddens[left], x[lcn]);
			_inputfilters[lcn].forward(cg, &_halfcells[lcn], &_inputgates[lcn]);

		/*	_forgetfilters[lcn].clear();
			_forgetfilters[lcn].resize(1);
			_forgetfilters[lcn][0].init(_outDim, -1);*/
			_forgetfilters[lcn][0].forward(cg, &_cells[left], &_forgetgates[lcn][0]);

			_cells[lcn].forward(cg, &_inputfilters[lcn], &_forgetfilters[lcn][0]);

			_halfhiddens[lcn].forward(cg, &_cells[lcn]);
			_outputgates[lcn].forward(cg, &_hiddens[left], x[lcn]);
			_hiddens[lcn].forward(cg, &_halfhiddens[lcn], &_outputgates[lcn]);
		}
		else { // lcn != i  && lcn != j
			assert(i != j);
			assert(lcn != i);
			_input_children_addition[lcn].forward(cg, &_hiddens[left], &_hiddens[right]);
			_inputgates[lcn].forward(cg, &_input_children_addition[lcn], x[lcn]);

			//_forgetgates[lcn].clear();
			//_forgetgates[lcn].resize(2);
			//_forgetgates[lcn][0].setParam(&_param->forget);
			//_forgetgates[lcn][0].setFunctions(&fsigmoid, &dsigmoid);
			//_forgetgates[lcn][0].init(_outDim, -1);
			_forgetgates[lcn][0].forward(cg, &_hiddens[left], x[lcn]);
		/*	_forgetgates[lcn][1].setParam(&_param->forget);
			_forgetgates[lcn][1].setFunctions(&fsigmoid, &dsigmoid);
			_forgetgates[lcn][1].init(_outDim, -1);*/
			_forgetgates[lcn][1].forward(cg, &_hiddens[right], x[lcn]);

			_halfcells[lcn].forward(cg, &_input_children_addition[lcn], x[lcn]);
			_inputfilters[lcn].forward(cg, &_halfcells[lcn], &_inputgates[lcn]);

		/*	_forgetfilters[lcn].clear();
			_forgetfilters[lcn].resize(2);*/

			/*_forgetfilters[lcn][0].init(_outDim, -1);*/
			_forgetfilters[lcn][0].forward(cg, &_cells[left], &_forgetgates[lcn][0]);
		
			/*_forgetfilters[lcn][1].init(_outDim, -1);*/
			_forgetfilters[lcn][1].forward(cg, &_cells[right], &_forgetgates[lcn][1]);

			_forgetfilters_addition[lcn].forward(cg, getPNodes(_forgetfilters[lcn], 2));

			_cells[lcn].forward(cg, &_inputfilters[lcn], &_forgetfilters_addition[lcn]);

			_halfhiddens[lcn].forward(cg, &_cells[lcn]);
			_outputgates[lcn].forward(cg, &_input_children_addition[lcn], x[lcn]);
			_hiddens[lcn].forward(cg, &_halfhiddens[lcn], &_outputgates[lcn]);
		}
	}

	inline void top_down_forward(Graph *cg, const vector<PNode>& x, const vector<TreeNode>& _tree_) {
		vector<TreeNode> tree = _tree_;

		assert(tree[0].parent == -1);
		int t = 0;

		_bucket.forward(cg, 0);
		_inputgates[t].forward(cg, &_bucket, x[t]);
		_halfcells[t].forward(cg, &_bucket, x[t]);
		_inputfilters[t].forward(cg, &_halfcells[t], &_inputgates[t]);
		_cells[t].forward(cg, &_inputfilters[t], &_bucket);
		_halfhiddens[t].forward(cg, &_cells[t]);
		_outputgates[t].forward(cg, &_bucket, x[t]);
		_hiddens[t].forward(cg, &_halfhiddens[t], &_outputgates[t]);



		vector<int> current;
		for (int i = 0; i < tree[0].children.size(); i++) {
			current.push_back(tree[0].children[i]);
		}
		
		
		while (!current.empty()) {
			for (int i = 0; i < current.size(); i++) {
				int t = current[i];
				int parent = tree[t].parent;
				

				
				_inputgates[t].forward(cg, &_hiddens[parent], x[t]);
				_outputgates[t].forward(cg, &_hiddens[parent], x[t]);
				_halfcells[t].forward(cg, &_hiddens[parent], x[t]);


				_forgetgates[t].clear();
				_forgetgates[t].resize(1);

				_forgetgates[t][0].setParam(&_param->forget);
				_forgetgates[t][0].setFunctions(&fsigmoid, &dsigmoid);
				_forgetgates[t][0].init(_outDim, -1);
				_forgetgates[t][0].forward(cg, &_hiddens[parent], x[t]);

				

				_inputfilters[t].forward(cg, &_halfcells[t], &_inputgates[t]);


				_forgetfilters[t].clear();
				_forgetfilters[t].resize(1);
				_forgetfilters[t][0].init(_outDim, -1);
				_forgetfilters[t][0].forward(cg, &_cells[parent], &_forgetgates[t][0]);

				_cells[t].forward(cg, &_inputfilters[t], &_forgetfilters[t][0]);

				_halfhiddens[t].forward(cg, &_cells[t]);

				_hiddens[t].forward(cg, &_halfhiddens[t], &_outputgates[t]);
			}

				
			vector<int> next;
			next.clear();
			for (int i = 0; i < current.size(); i++) {
				for (int j = 0; j < tree[current[i]].children.size(); j++) {
					next.push_back(tree[current[i]].children[j]);
				}
			}
			current.clear();
			current = next;
		}

	}


	inline void top_down_forward(Graph *cg, const vector<PNode>& x, const vector<TreeNode>& _tree_, int i, int j) {
		vector<TreeNode> tree = _tree_;
		assert(tree[0].parent == -1);
		malloc_path_nodes(i, j, tree, false);

		int lcn = find_least_common_node(tree, i, j);


		lcn_cell_init.forward(cg, 0);
		lcn_hidden_init.forward(cg, 1);
		_inputgates[lcn].forward(cg, &lcn_hidden_init, x[lcn]);
		_forgetgates[lcn][0].forward(cg, &lcn_hidden_init, x[lcn]);
		_outputgates[lcn].forward(cg, &lcn_hidden_init, x[lcn]);
		_halfcells[lcn].forward(cg, &lcn_hidden_init, x[lcn]);

		_inputfilters[lcn].forward(cg, &_halfcells[lcn], &_inputgates[lcn]);
		_forgetfilters[lcn][0].forward(cg, &lcn_cell_init, &_forgetgates[lcn][0]);
		_cells[lcn].forward(cg, &_inputfilters[lcn], &_forgetfilters[lcn][0]);
		_halfhiddens[lcn].forward(cg, &_cells[lcn]);
		_hiddens[lcn].forward(cg, &_halfhiddens[lcn], &_outputgates[lcn]);

	/*	_bucket.forward(cg, 0);
		_inputgates[lcn].forward(cg, &_bucket, x[lcn]);
		_halfcells[lcn].forward(cg, &_bucket, x[lcn]);
		_inputfilters[lcn].forward(cg, &_halfcells[lcn], &_inputgates[lcn]);
		_cells[lcn].forward(cg, &_inputfilters[lcn], &_bucket);
		_halfhiddens[lcn].forward(cg, &_cells[lcn]);
		_outputgates[lcn].forward(cg, &_bucket, x[lcn]);
		_hiddens[lcn].forward(cg, &_halfhiddens[lcn], &_outputgates[lcn]);*/
		

		if (i == j) {
			return;
		}

		if (i != lcn) {
			vector<int> left_path;
			find_sp(tree, i, lcn, left_path);

			for (int m = left_path.size() - 2; m >= 0; m--) {
				int current = left_path[m];
				int pre = left_path[m + 1];

				_inputgates[current].forward(cg, &_hiddens[pre], x[current]);

				/*_forgetgates[current].clear();
				_forgetgates[current].resize(1);
				_forgetgates[current][0].setParam(&_param->forget);
				_forgetgates[current][0].setFunctions(&fsigmoid, &dsigmoid);
				_forgetgates[current][0].init(_outDim, -1);*/
				_forgetgates[current][0].forward(cg, &_hiddens[pre], x[current]);

				_halfcells[current].forward(cg, &_hiddens[pre], x[current]);
				_inputfilters[current].forward(cg, &_halfcells[current], &_inputgates[current]);

				/*_forgetfilters[current].clear();
				_forgetfilters[current].resize(1);
				_forgetfilters[current][0].init(_outDim, -1);*/
				_forgetfilters[current][0].forward(cg, &_cells[pre], &_forgetgates[current][0]);

				_cells[current].forward(cg, &_inputfilters[current], &_forgetfilters[current][0]);

				_halfhiddens[current].forward(cg, &_cells[current]);
				_outputgates[current].forward(cg, &_hiddens[pre], x[current]);
				_hiddens[current].forward(cg, &_halfhiddens[current], &_outputgates[current]);
			}
		}
		
		if (j != lcn) {
			vector<int> right_path;
			find_sp(tree, j, lcn, right_path);

			for (int m = right_path.size() - 2; m >= 0; m--) {
				int current = right_path[m];
				int pre = right_path[m + 1];

			//	cout << "pre: " << pre << endl;
			//	cout << "current: " << current << endl;
			//	cout << endl;
				_inputgates[current].forward(cg, &_hiddens[pre], x[current]);

			/*	_forgetgates[current].clear();
				_forgetgates[current].resize(1);
				_forgetgates[current][0].setParam(&_param->forget);
				_forgetgates[current][0].setFunctions(&fsigmoid, &dsigmoid);
				_forgetgates[current][0].init(_outDim, -1);*/
				_forgetgates[current][0].forward(cg, &_hiddens[pre], x[current]);

				_halfcells[current].forward(cg, &_hiddens[pre], x[current]);
				_inputfilters[current].forward(cg, &_halfcells[current], &_inputgates[current]);

			/*	_forgetfilters[current].clear();
				_forgetfilters[current].resize(1);
				_forgetfilters[current][0].init(_outDim, -1);*/
				_forgetfilters[current][0].forward(cg, &_cells[pre], &_forgetgates[current][0]);

				_cells[current].forward(cg, &_inputfilters[current], &_forgetfilters[current][0]);

				_halfhiddens[current].forward(cg, &_cells[current]);
				_outputgates[current].forward(cg, &_hiddens[pre], x[current]);
				_hiddens[current].forward(cg, &_halfhiddens[current], &_outputgates[current]);
			}
		}
	}
};

#endif
