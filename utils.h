#ifndef UTILS_H
#define UTILS_H

#endif // UTILS_H

#pragma once

#include <list>

class ExpMovingAverage {
private:
    double alpha; // [0;1] less = more stable, more = less stable
    double oldValue;
    bool unset;
public:
    ExpMovingAverage() {
        this->alpha = 0.2;
        unset = true;
    }

    void clear() {
        unset = true;
    }

    void add(double value) {
        if (unset) {
            oldValue = value;
            unset = false;
        }
        double newValue = oldValue + alpha * (value - oldValue);
        oldValue = newValue;
    }

    double get() {
        return oldValue;
    }
};

