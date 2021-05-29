import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Initializing probality dictionary to contain person with its probablity
    probability = {}

    # Looping for every person in people dict.
    for person in people:

        # Checking if mother exists
        if people[person]['mother'] == None:

            # Getting gene info of person
            if person in one_gene:
                gene = 1
            elif person in two_genes:
                gene = 2
            else:
                gene = 0
            
            # Getting gene probablity from PROBS according to the person's gene query
            gene_prob = PROBS["gene"][gene]

            # Getting trait probablity from PROBS according to the person's gene query
            if person in have_trait:
                trait_prob = PROBS["trait"][gene][True]
            else:
                trait_prob = PROBS["trait"][gene][False]
            
            # Adding gene and trait prob. product to probablity
            probability[person] = gene_prob * trait_prob

        else:

            # Getting the probablity of gene from mother according to mother's gene info
            if people[person]['mother'] in one_gene:
                from_mother = 0.50
            elif people[person]['mother'] in two_genes:
                from_mother = 1 - PROBS['mutation']
            else:
                from_mother = PROBS['mutation']
            
            # Getting the probablity of gene from father according to mother's gene info
            if people[person]['father'] in one_gene:
                from_father = 0.50
            elif people[person]['father'] in two_genes:
                from_father = 1.00 - PROBS['mutation']
            else:
                from_father = PROBS['mutation']

            # Getting the person's gene info
            if person in one_gene:
                # For 1 gene: (none from mother * 1 from father) + ( 1 from mother * none from father)
                gene_prob = ((1 - from_mother) * from_father) + ((1 - from_father) * from_mother)
                gene = 1
            elif person in two_genes:
                # For 2 gene: 1 from mother * 1 from father
                gene_prob = from_mother * from_father
                gene = 2
            else:
                # For 2 gene: none from mother * none from father
                gene_prob = (1 - from_father) * (1 - from_mother)
                gene = 0
            
            # Getting the person's trait info
            if person in have_trait:
                trait_prob = PROBS["trait"][gene][True]
            else:
                trait_prob = PROBS["trait"][gene][False]
            
            # Adding gene and trait prob. product to probablity
            probability[person] = gene_prob * trait_prob
    
    # Getting the final Joint Probablity by multiplying all probablities in probablity dict.
    result = 1
    for prob in probability:
        result *= probability[prob]
    
    return result


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:

        # Geting gene info
        if person in one_gene:
            gene = 1
        elif person in two_genes:
            gene = 2
        else:
            gene = 0

        # Adding p to appropriate fields
        probabilities[person]["gene"][gene] += p
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    for person in probabilities:

        # Normalizing genes
        genes = []
        for i in range(3):
            genes.append(probabilities[person]["gene"][i])
        normalize = [float(i)/sum(genes) for i in genes]
        for i in range(3):
            probabilities[person]["gene"][i] = normalize[i]
        
        # Normalizing traits
        traits = []
        traits.append(probabilities[person]["trait"][True])
        traits.append(probabilities[person]["trait"][False])
        normalize = [float(i)/sum(traits) for i in traits]
        probabilities[person]["trait"][True] = normalize[0]
        probabilities[person]["trait"][False] = normalize[1]

if __name__ == "__main__":
    main()
