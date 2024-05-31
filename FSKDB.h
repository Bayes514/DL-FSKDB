#pragma once
#include "incrementalLearner.h"
#include "xxxyDist.h"
#include "xxyDist.h"
#include <limits>
#include "crosstab.h"



class FSKDB: public IncrementalLearner {
public:
	FSKDB();
	FSKDB(char* const *& argv, char* const * end);
	~FSKDB(void);

	void reset(InstanceStream &is);   ///< reset the learner prior to training
	void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
	void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
	void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
	void getCapabilities(capabilities &c);
	virtual void classify(const instance &inst, std::vector<double> &classDist);
	//void getCMI(xxyDist &dist,std::vector<std::vector<double> > & CMI);
	double getSCMI(xxyDist &dist, const instance& inst, CategoricalAttribute x1,CategoricalAttribute x2);
	double getSCMI2(xxyDist &dist, const instance& inst, CategoricalAttribute x1,CategoricalAttribute x2);
    //double getSCMI2(xxxyDist &Dist, const instance& inst, CategoricalAttribute x1,CategoricalAttribute x2, CategoricalAttribute x3);
private:
	unsigned int noCatAtts_;          ///< the number of categorical attributes.
	unsigned int noClasses_;                          ///< the number of classes
	InstanceStream* instanceStream_;
	xxyDist xxyDist_;
	xxxyDist xxxyDist_;
	//unsigned int pass_;
    unsigned int k_;
    //std::vector<std::vector<double> >  CMI;
    std::vector<CategoricalAttribute> attsort;
    std::vector<std::vector<CategoricalAttribute> > candsort;
	bool trainingIsFinished_; ///< true iff the learner is trained
	const static CategoricalAttribute NOPARENT = 0xFFFFFFFFUL; //使用printf("%d",0xFFFFFFFFUL);输出是-1 cannot use std::numeric_limits<categoricalAttribute>::max() because some compilers will not allow it here
};
