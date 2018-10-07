#ifndef DROPOUT_H_
#define DROPOUT_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

//jz
inline void getMask(Tensor1D &mask, dtype drop_value) {
	int dimension = mask.dim;
	if (drop_value > 0) {
		for (int idx = 0; idx < dimension; idx++) {
			dtype r = (dtype(rand()) / RAND_MAX) * (1.0 - 0.0) + 0.0;
			mask[idx] = (r > drop_value) ? 1.0 / (1.0 - drop_value) : 0.0;
		}
	}
}

inline void getEmbMask(Tensor1D &wmask, Tensor1D &tmask, dtype drop_value) {
	int wdim = wmask.dim;
	int tdim = tmask.dim;
	dtype r;
	if (drop_value > 0) {
		r = (dtype(rand()) / RAND_MAX) * (1.0 - 0.0) + 0.0;
		dtype wm = (r > drop_value) ? 1.0 : 0.0;
		r = (dtype(rand()) / RAND_MAX) * (1.0 - 0.0) + 0.0;
		dtype tm = (r > drop_value) ? 1.0 : 0.0;
		
		dtype scale = 3.0 / (2.0 * wm + tm + 1e-12);
		wm *= scale;
		tm *= scale;
		
		for (int idx = 0; idx < wdim; idx++) {
			wmask[idx] = wm;
			
		}
		for (int idx = 0; idx < tdim; idx++) {
			tmask[idx] = tm;
			
		}
	}
}


class DropNode : public Node {
public:
	PNode in;

	DropNode() : Node() {
		in = NULL;
		node_type = "dropout";
	}
	~DropNode() {
		in = NULL;
	}
	inline void clearValue() {
		Node::clearValue();
		in = NULL;
	}

public:
	void forward(Graph *cg, PNode x) {
		in = x;
		degree = 0;
		in->addParent(this);
		cg->addNode(this);
	}

public:
	inline void setMask(const Tensor1D &mask) {
		assert(drop_mask.dim == mask.dim);
		drop_mask.vec() = mask.vec();
	}
	inline void compute(bool bTrain) {
		if (drop_value > 0 && bTrain) {
			val.vec() = in->val.vec() * drop_mask.vec();
		}
		else {
			val.vec() = in->val.vec();
		}
	}
	inline void backward() {
		if (drop_value > 0) {
			in->loss.vec() += loss.vec() * drop_mask.vec();
		}
		else {
			in->loss.vec() += loss.vec();
		}
	}

public:
	inline PExecute generate(bool bTrain);

	// better to rewrite for deep understanding
	inline bool typeEqual(PNode other) {
		return Node::typeEqual(other);
	}
};

class DropExecute :public Execute {
public:
	bool bTrain;
public:
	inline void forward() {
		int count = batch.size();
		for (int idx = 0; idx < count; idx++) {
			DropNode* ptr = (DropNode*)batch[idx];
			ptr->compute(bTrain);
		}
	}

	inline void backward() {
		int count = batch.size();
		for (int idx = 0; idx < count; idx++) {
			DropNode* ptr = (DropNode*)batch[idx];
			ptr->backward();
		}
	}
};

inline PExecute DropNode::generate(bool bTrain) {
	DropExecute* exec = new DropExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
}


#endif
