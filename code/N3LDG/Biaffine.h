#ifndef Biaffine_H_
#define Biaffine_H_

/*
*  Biaffine_H_.h:
*  a simple feed forward neural operation, binary input.
*
*  Created on: June 11, 2017
*      Author: yue zhang (suda)
*/


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class BiaffineParams {
  public:
    vector<Param> W;
    Param b;
    bool bUseB;
    int classDim;

  public:
    BiaffineParams() {
        bUseB = true;
        classDim = 0;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        for (int i = 0; i < classDim; i++) {
            ada.addParam(&W[i]);
        }
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    inline void initial(int nISize1, int nISize2, bool useB = true, int classDims = 1) {
        classDim = classDims;
        W.resize(classDim);
        for (int i = 0; i < classDim; i++) {
            W[i].initial_constant(nISize1, nISize2);
        }
        bUseB = useB;
        if (bUseB) {
            b.initial_constant(classDim, 1);
        }
    }

    inline void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        for (int i = 0; i < classDim; i++)
            W[i].save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    inline void load(std::ifstream &is) {
        is >> bUseB;
        for (int i = 0; i < classDim; i++)
            W[i].load(is);
        if (bUseB) {
            b.load(is);
        }
    }
};

class SBiaffineParam {
public:
	Param U;
	Param b;
	bool bUseB;

public:
	SBiaffineParam() {
		bUseB = true;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		ada.addParam(&U);
		if (bUseB) {
			ada.addParam(&b);
		}
	}

	inline void initial(int nISize1, int nISize2, bool useB = true) {
		U.initial(nISize1, nISize2);
		bUseB = useB;
		if (bUseB) {
			b.initial_constant(1, 1);
		}
	}

	inline void save(std::ofstream &os) const {
		os << bUseB << std::endl;
		U.save(os);
		if (bUseB) {
			b.save(os);
		}
	}

	inline void load(std::ifstream &is) {
		is >> bUseB;
		U.load(is);
		if (bUseB) {
			b.load(is);
		}
	}
};


class BiaffineNode : public Node {
  public:
    vector<PNode> in1, in2;
    Tensor2D x1, x2;//concat nodes in in1 by y

    int expandIn1, expandIn2;
    BiaffineParams* param;
    int nSize;
    int classDim;
    int inDim1, inDim2;

    vector<Tensor2D> vals;
    vector<Tensor2D> losses;
    vector<Tensor2D> y1;

  public:
    BiaffineNode() : Node() {
        nSize = 0;
        classDim = 0;
        in1.clear();
        in2.clear();
        vals.clear();
        losses.clear();
        param = NULL;
        node_type = "biaffine";
    }

    inline void setParam(BiaffineParams* paramInit, int expandIns1, int expandIns2) {
        param = paramInit;
        classDim = paramInit->classDim;
        expandIn1 = expandIns1;
        expandIn2 = expandIns2;
    }

    inline void clearValue() {
        Node::clearValue();
        in1.clear();
        in2.clear();
        vals.clear();
        losses.clear();
		y1.clear();
		x1.zero();
		x2.zero();
    }

    inline void init(int dim) {
        this->dim = dim;
        vals.resize(classDim);
        losses.resize(classDim);
        for (int i = 0; i < classDim; i++) {
            vals[i].init(dim, dim);
            losses[i].init(dim, dim);
        }
        parents.clear();
    }

  public:
    void forward(Graph *cg, vector<PNode> x1, vector<PNode> x2) {
        assert(x1.size() == x2.size());
        nSize = x1.size();
        for (int i = 0; i < nSize; i++) {
            in1.push_back(x1[i]);
            in2.push_back(x2[i]);
        }
        degree = in1.size() + in2.size();
        for (int i = 0; i < nSize; i++) {

            in1[i]->parents.push_back(this);
			
            in2[i]->parents.push_back(this);
        }
        cg->addNode(this);
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;
        BiaffineNode* conv_other = (BiaffineNode*)other;
        if (param == conv_other->param) {
            return true;
        } else
            return false;
    }
  public:
    inline void compute() {
        inDim1 = in1[0]->dim;
        inDim2 = in2[0]->dim;
		x1.init(inDim1 + expandIn1, nSize);
		x2.init(inDim2 + expandIn2, nSize);
       /* x1.init(inDim1 + (expandIn1 ? 1 : 0), nSize);
        x2.init(inDim2 + (expandIn2 ? 1 : 0), nSize);*/
        y1.resize(classDim);
        for (int i = 0; i < classDim; i++) {
			y1[i].init(nSize, inDim2 + expandIn2);
           /* y1[i].init(nSize, inDim2 + (expandIn2 ? 1 : 0));*/
        }
        for (int i = 0; i < nSize; ++i) {
			for (int j = 0; j < inDim1; j++) {
				x1[i][j] = in1[i]->val[j];
			}
			
			for (int k = inDim1; k < inDim1 + expandIn1; k++) {
				x1[i][k] = 1; 
			}
            /*if (expandIn1)
                x1[i][inDim1] = 1;*/

			for (int j = 0; j < inDim2; j++) {
				x2[i][j] = in2[i]->val[j];
			}

			for (int k = inDim2; k < inDim2 + expandIn2; k++) {
				x2[i][k] = 1;
			}

           /* if (expandIn2)
                x2[i][inDim2] = 1;*/
        }


        for (int i = 0; i < classDim; i++) {
            y1[i].mat() = x1.mat().transpose() * param->W[i].val.mat();
            vals[i].mat() = y1[i].mat() * x2.mat();
            if (param->bUseB) {
                for (int idx = 0; idx < nSize; idx++)
                    for (int idy = 0; idy < nSize; idy++)
                        vals[i].mat()(idx, idy) += param->b.val.mat()(i, 0);
            }
        }
    }

    inline void backward() {
        vector<Tensor2D> lx1, lx2, ly1;
        lx1.resize(classDim);
        lx2.resize(classDim);
        ly1.resize(classDim);
        for (int i = 0; i < classDim; i++) {
            lx1[i].init(inDim1 + expandIn1, nSize);
            lx2[i].init(inDim2 + expandIn2, nSize);
			ly1[i].init(nSize, inDim2 + expandIn2);
        }

        for (int i = 0; i < classDim; i++) {
            lx2[i].mat() = y1[i].mat().transpose() * losses[i].mat();
            ly1[i].mat() = losses[i].mat() * x2.mat().transpose();
            lx1[i].mat() = param->W[i].val.mat() * ly1[i].mat().transpose();
            param->W[i].grad.mat() += x1.mat() * ly1[i].mat();
        }

        if (param->bUseB) {
            for (int i = 0; i < classDim; i++) {
                for (int idx = 0; idx < nSize; idx++) {
                    for (int idy = 0; idy < nSize; idy++)
                        param->b.grad.v[i] += losses[i][idx][idy];
                }
            }
        }
        for (int i = 0; i < classDim; i++) {
            for (int idx = 0; idx < nSize; idx++) {
                for (int idy = 0; idy < inDim1; idy++) {
                    in1[idx]->loss[idy] += lx1[i][idx][idy];
                }
                for (int idy = 0; idy < inDim2; idy++) {
                    in2[idx]->loss[idy] += lx2[i][idx][idy];
                }
            }
        }
    }
};

class BiaffineExecute :public Execute {
  public:
    inline void  forward() {
        int count = batch.size();

        for (int idx = 0; idx < count; idx++) {
            BiaffineNode* ptr = (BiaffineNode*)batch[idx];
            ptr->compute();
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            BiaffineNode* ptr = (BiaffineNode*)batch[idx];
            ptr->backward();
        }
    }
};

inline PExecute BiaffineNode::generate(bool bTrain) {
    BiaffineExecute* exec = new BiaffineExecute();
    exec->batch.push_back(this);
    return exec;
};

class SBiaffineNode : public Node {
public:
	PNode in1, in2;
	Tensor1D x1, x2;
	Tensor2D tmp;

	int expandIn1, expandIn2;
	SBiaffineParam* param;
	/*int nSize;*/
	int inDim;

	//vector<Tensor2D> vals;
	//vector<Tensor2D> losses;
	//vector<Tensor2D> y1;

public:
	SBiaffineNode() : Node() {
		/*nSize = 0;*/
		in1 = NULL;
		in2 = NULL;
		param = NULL;
		node_type = "s-biaffine";
	}

	inline void setParam(SBiaffineParam* paramInit, int expandIns1, int expandIns2) {
		param = paramInit;
		expandIn1 = expandIns1;
		expandIn2 = expandIns2;
	}

	inline void clearValue() {
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
		x1.zero();
		x2.zero();
		tmp.zero();
		/*vals.clear();
		losses.clear();*/
		/*y1.clear();
		x1.zero();
		x2.zero();*/
	}

	//inline void init(int dim) {
	//	this->dim = dim;
	//	vals.resize(classDim);
	//	losses.resize(classDim);
	//	for (int i = 0; i < classDim; i++) {
	//		vals[i].init(dim, dim);
	//		losses[i].init(dim, dim);
	//	}
	//	parents.clear();
	//}

public:
	void forward(Graph *cg, PNode x1, PNode x2) {
		in1 = x1;
		in2 = x2;
		degree = 0;
		in1->addParent(this);
		in2->addParent(this);
		cg->addNode(this);

		/*assert(x1.size() == x2.size());
		nSize = x1.size();
		for (int i = 0; i < nSize; i++) {
		in1.push_back(x1[i]);
		in2.push_back(x2[i]);
		}
		degree = in1.size() + in2.size();
		for (int i = 0; i < nSize; i++) {
		in1[i]->parents.push_back(this);
		in2[i]->parents.push_back(this);
		}
		cg->addNode(this);*/
	}

public:
	inline PExecute generate(bool bTrain);

	// better to rewrite for deep understanding
	inline bool typeEqual(PNode other) {
		bool result = Node::typeEqual(other);
		if (!result) return false;
		SBiaffineNode* conv_other = (SBiaffineNode*)other;
		if (param == conv_other->param) {
			return true;
		}
		else
			return false;
	}
public:
	inline void compute() {
		assert(in1->dim == in2->dim);
		inDim = in1->dim;
		x1.init(inDim + expandIn1);
		x2.init(inDim + expandIn2);
		for (int idx = 0; idx < inDim; idx++)
		{
			x1[idx] = in1->val[idx];
			x2[idx] = in2->val[idx];
		}
		for (int k = inDim; k < inDim + expandIn1; k++) {
			x1[k] = 1;
		}
		for (int k = inDim; k < inDim + expandIn2; k++) {
			x2[k] = 1;
		}

		tmp.init(1, inDim + expandIn2);

		//cout << "dim: " << x1.dim << endl;
        //cout << "row: " << param->U.val.row << "col: " << param->U.val.col << endl;
		//cout << "tmp row: " << tmp.row << "tmp col: " << tmp.col << endl;
		tmp.mat() = x1.mat().transpose() * param->U.val.mat();
		
		val.mat() = tmp.mat() * x2.mat();

		if (param->bUseB) {
			val.mat() += param->b.val.mat();
		}
	}

	inline void backward() {
		Tensor1D lx1, lx2;
		Tensor2D ltmp;

		lx1.init(inDim + expandIn1);
		lx2.init(inDim + expandIn2);
		ltmp.init(1, inDim + expandIn2);

		lx2.mat() = tmp.mat().transpose() * loss.mat();
		ltmp.mat() = loss.mat() * x2.mat().transpose();
		lx1.mat() = param->U.val.mat() * ltmp.mat().transpose();
		param->U.grad.mat() += x1.mat() * ltmp.mat();

		if (param->bUseB) {
			param->b.grad.mat() += loss.mat();
		}

		for (int idx = 0; idx < inDim; idx++) {
			in1->loss[idx] += lx1[idx];
			in2->loss[idx] += lx2[idx];
		}
	}
};

class SBiaffineExecute :public Execute {
public:
	inline void  forward() {
		int count = batch.size();

		for (int idx = 0; idx < count; idx++) {
			SBiaffineNode* ptr = (SBiaffineNode*)batch[idx];
			ptr->compute();
		}
	}

	inline void backward() {
		int count = batch.size();
		for (int idx = 0; idx < count; idx++) {
			SBiaffineNode* ptr = (SBiaffineNode*)batch[idx];
			ptr->backward();
		}
	}
};

inline PExecute SBiaffineNode::generate(bool bTrain) {
	SBiaffineExecute* exec = new SBiaffineExecute();
	exec->batch.push_back(this);
	return exec;
};


#endif /* Biaffine_H_ */
