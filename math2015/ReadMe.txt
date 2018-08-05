1. There are three directories in the current folder, and each directory contains a piece of data used in our paper as follows:
FrcSub-----------------The public dataset, widely used in cognitive modelling (e.g., [Tatsuoka, 1984; Junker and Sijtsma, 2001; DeCarlo, 2010]), is made up of test responses (right or wrong, coded to 1 or 0) of examinees on Fraction-Substraction problems.
Math1&Math2------------The private datasets we used include two final math examination results (scores of each examinee on each problem) of a high school.

2. There are four files in each directory as follows:
data.txt---------------The responses or normalized scores (which are scaled in range [0,1] by dividing full scores of each problem) of each examinee on each problems, and a row denotes an examinee while a column stands for a problem.
qnames.txt-------------The detailed names or meanings of related specific skill.
q.txt------------------The indicator matrix of relationship between problems and skills, which derives from experienced education experts. And a row represents a problem while a column for a skill. E.g., problem i requires skill k if entry(i, k) equals to 1 and vice versa.
problemdesc.txt--------The description of each problem, including the problem type (objective or subjective) and full scores of each problem (set to 1 for all the problems in FrcSub dataset).

3. Besides, there is one more file in Math1 and Math2 directories.
rawdata.txt------------The raw unnormalized scores of the Math1 and Math2 datasets.

4. For better understanding, we give two examples of how to use the datasets in the file "Example.txt" in the current folder.

5. And if you intend to use the two private datasets (called Math dataset) for any exploratory analysis, please refer to the Terms of Use, which is decribed in the file "TermsOfUse.txt" in detail.

