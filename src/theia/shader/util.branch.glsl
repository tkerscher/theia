#ifndef _INCLUDE_UTIL_BRANCH
#define _INCLUDE_UTIL_BRANCH

//subgroupAll() may allow the driver to create conditional branching,
//i.e. if all invocation of a subgroup share the same bool, only execute one
//branch. Not guarenteed to work.

#define CHECK_BRANCH(x) subgroupAll(x) || (!subgroupAll(!(x)) && x)

#endif