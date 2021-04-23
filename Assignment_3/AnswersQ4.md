# Question 4

## Time Complexity
- M : Number of features
- N : Number of Samples
- K : Number of classes
- E : Epochs

### Training
- ```O(M*N*K*E)```
- Here number of classes also in time complexity because of softmax 
### Prediction
- ```O(N*M*K)```

## Space Complexity
### Training
- ```O(M*N+M*K+N*K)```if we count input also into the space complexity.
- Else the input matrix we can remove from the space
- Whereas if we not count the space used by input, then only the space taken by multiplication of input with parameters is space complexity. ``` O(N*K+M*K) ```
### Prediction
- ```O(M*N+M*K+N*K)```if we count input also into the space complexity.
- Whereas if we not count the space used by input, then only the space taken by multiplication of input with parameters is space complexity. ``` O(N*K) ```