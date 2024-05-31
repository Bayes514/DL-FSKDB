#include "FSKDB.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>
#include "queue"


class miCmpClass {
public:
  miCmpClass(std::vector<float> *m) {
    mi = m;
  }

  bool operator() (CategoricalAttribute a, CategoricalAttribute b) {
    return (*mi)[a] > (*mi)[b];
  }

private:
  std::vector<float> *mi;
};


struct cmicmp {
	bool operator ()( std::pair<int,double> a, std::pair<int,double> b) {
		return a.second < b.second;
	}
};


FSKDB::FSKDB():trainingIsFinished_(false)
{
}

FSKDB::FSKDB(char*const*& argv, char*const* end) :trainingIsFinished_(false)
{
    name_ = "FSKDB";
     k_ = 1;
    while (argv != end)
    {
        if (*argv[0] != '-')
        {
            break;
        } else if (argv[0][1] == 'k')
        {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else
        {
            break;
        }
        name_ += argv[0];

        ++argv;
    }
}

FSKDB::~FSKDB(void)
{
}

void FSKDB::reset(InstanceStream &is)
{
     instanceStream_ = &is;
    noCatAtts_ = is.getNoCatAtts();
    noClasses_ = is.getNoClasses();
     k_ = min(k_, noCatAtts_ - 1);
    trainingIsFinished_ = false;
    //CMI.resize(noCatAtts_);
    attsort.resize(noCatAtts_);
    candsort.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        //CMI[a].resize(noCatAtts_);
        candsort[a].clear();
        attsort[a]=a;
    }

    xxyDist_.reset(is);
    xxxyDist_.reset(is);
    //pass_ = 1;
}

void FSKDB::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void FSKDB::initialisePass()
{
}

void FSKDB::train(const instance &inst)
{
     if (k_ == 1)
    {
        xxyDist_.update(inst);
    }
     else if(k_==2)
    {
         xxxyDist_.update(inst);
    }
}

//D(x1;x2|Y)
double FSKDB::getSCMI(xxyDist &dist, const instance& inst, CategoricalAttribute x1,CategoricalAttribute x2)
{
    const double totalCount = dist.xyCounts.count;
    float m = 0.0;
    CatValue v1 = inst.getCatVal(x1);
    CatValue v2 = inst.getCatVal(x2);
    for (CatValue y = 0; y < dist.getNoClasses(); y++)
    {
        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
        double temp=0.0;
        if (x1x2y)
        {
            temp += (x1x2y / totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                            (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                            dist.xyCounts.getCount(x2, v2, y)));
            temp=fabs(temp);
        }
        m+=temp;
    }
    return m;
}

//D(x1;Y|x2)
double FSKDB::getSCMI2(xxyDist &dist, const instance& inst, CategoricalAttribute x1,CategoricalAttribute x2)
{
    const double totalCount = dist.xyCounts.count;
    float m = 0.0;
    CatValue v1 = inst.getCatVal(x1);
    CatValue v2 = inst.getCatVal(x2);
    for (CatValue y = 0; y < dist.getNoClasses(); y++)
    {
        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
        double temp=0.0;
        if (x1x2y)
        {
            temp += (x1x2y / totalCount) * log2(dist.xyCounts.getCount(x2,v2) * x1x2y /
                            (static_cast<double> (dist.getCount(x1,v1,x2,v2)) *
                            dist.xyCounts.getCount(x2, v2, y)));
            temp=fabs(temp);
        }
        m+=temp;
    }
    return m;
}

void FSKDB::classify(const instance &inst, std::vector<double> &classDist)
{
   //printf("--classify--\n");
    if(k_ == 2)
        xxyDist_ = xxxyDist_.xxyCounts;
    std::vector<std::vector<CategoricalAttribute> > parents;
    parents.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        parents[a].clear();
        //printf("%d\n",attsort[a]);
    }


    std::vector<bool>  isatt_ (noCatAtts_, true);


    for (int i=0; i<attsort.size(); ++i)
    {
        //printf("%d\n",a);
        CategoricalAttribute a = attsort[i];
        for(int j=0;  isatt_[a] && j<candsort[a].size() && parents[a].size() < k_;++j)
        {
                CategoricalAttribute candpa = candsort[a][j];
                if(!isatt_[candpa])
                    continue;
                double lcmi = getSCMI(xxyDist_, inst, a, candpa);
                double lcmi2 = getSCMI2(xxyDist_,inst,a,candpa);
                if(lcmi > 0)
                {
                    if(lcmi2 > 0)
                        parents[a].push_back(candpa);
                    else
                    {
                        isatt_[a]=false;
                        //printf("lcmi2:%f\n",lcmi2);
                    }
                }
        }
    }


    int attcnt = 0;
    int decnt = 0;
    for(unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        if (isatt_[x1])
        {
            attcnt ++;
            //printf("%d:",x1);
            /*
            for(int k=0;k<parents[x1].size();++k)
                decnt ++;
            */
            decnt += parents[x1].size();
                //printf("%d,",parents[x1][k]);
            //printf("\n");
        }
    }
    printf("used att:%d, edges:%d\n",attcnt,decnt);


    /*
    int esum = 0;
    for(unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
       esum += parents[x1].size();
    }
    printf("%d\n",esum);
    */

    for (CatValue y = 0; y < noClasses_; y++)
    {
        classDist[y] =  xxyDist_.xyCounts.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        if(isatt_[x1])
        {
            if (parents[x1].size() == 0)
            {
                for (CatValue y = 0; y < noClasses_; y++)
                {
                    classDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
                }
            }
            else if(parents[x1].size() == 1)
            {
                const CategoricalAttribute parent = parents[x1][0];
                for (CatValue y = 0; y < noClasses_; y++)
                {
                    classDist[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,
                            inst.getCatVal(parent), y);
                }
            }
            else if(parents[x1].size() == 2)
            {
                const CategoricalAttribute parent1 = parents[x1][0];
                const CategoricalAttribute parent2 = parents[x1][1];
                for (CatValue y = 0; y < noClasses_; y++)
                {
                    classDist[y] *= xxxyDist_.p(x1,inst.getCatVal(x1),parent1,inst.getCatVal(parent1),
                                        parent2,inst.getCatVal(parent2),y);
                }
            }
        }
    }

    normalise(classDist);
}

void FSKDB::finalisePass()
{
    assert(trainingIsFinished_ == false);
    if(k_ == 2)
        xxyDist_ = xxxyDist_.xxyCounts;

    std::vector<float> mi;
    getMutualInformation(xxyDist_.xyCounts, mi);
    miCmpClass cmp(&mi);
    std::sort(attsort.begin(), attsort.end(), cmp);

    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(xxyDist_, cmi);

    for (int i=1; i<noCatAtts_; ++i)
    {
        std::priority_queue<std::pair<int,double>, std::vector<std::pair<int,double> >, cmicmp> qsort;
        for(int j=0;j<i;++j)
            qsort.push(std::make_pair(attsort[j],cmi[attsort[i]][attsort[j]]));
        while(!qsort.empty())
        {
            CategoricalAttribute att=qsort.top().first;
            qsort.pop();
            candsort[attsort[i]].push_back(att);
        }
    }


    /*
    for (int i= 0; i < noCatAtts_ ; i++)
    {
        CategoricalAttribute att = attsort[i];
        printf("%d:%f, %d\n",att,mi[att],candsort[att].size());
    }

     for (int i= 0; i < noCatAtts_ ; i++)
    {
        CategoricalAttribute att = attsort[i];
        printf("--------%d---------\n",att);
        for(int j=0;j<candsort[att].size();++j)
            printf("%d,%f\n",candsort[att][j],cmi[att][candsort[att][j]]);
    }
    */
    trainingIsFinished_ = true;
}

bool FSKDB::trainingIsFinished()
{
    return trainingIsFinished_;
}
