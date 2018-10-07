/**
 * Implements the feature class
 * @file Feature.h
 * @author Mihai Surdeanu
 */

#ifndef EGSTRA_FEATURE_H
#define EGSTRA_FEATURE_H

namespace deeper {

  class Feature {
  private:
    /** The index of this feature in the feature dictionary */
    int mIndex;

    /**
     * The string value of this feature
     * This is instantiated only for the token local features,
     *   which are expanded to generate the token context features
     */
    std::string mValue;

  public:
    Feature(int i) { mIndex = i; }

    Feature(int i, const std::string & v) { mIndex = i; mValue = v; }

    const std::string & getValue() const { return mValue; }
    int getIndex() const { return mIndex; }
  };

}

#endif /* EGSTRA_FEATURE_H */
