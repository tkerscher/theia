#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rest of the code interacts with
//hit respnoses.
//
//Note that hit responses consume HitItem and thus depend on the used ray model.

/**
 * Consumes the given hit item to produce a detector response.
*/
void response(
    HitItem item,               ///< Item to process
    uint idx, inout uint dim    ///< RNG state
);
