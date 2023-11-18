function [conTrainX,conTrainC] = trainingDataSelection(archiveX,archiveC,numTrain)
    n = size(archiveX,2);
    classNum = round(numTrain/2);
    [archiveFeaInd,archiveInfeaInd] = judgeFeasible(archiveC);
    % boundary training data set selection
    if ~isempty(archiveInfeaInd)
        feaCanX = archiveX(archiveFeaInd,:);
        infeaCanX = archiveX(archiveInfeaInd,:);
        if isempty(archiveFeaInd) %ȫ�ǲ����н⣬ѡ���õ�300�����������ع�
            conTrainX = archiveX(1:numTrain,:);
            conTrainC = archiveC(1:numTrain,:);
            
        elseif ~isempty(archiveFeaInd) && length(archiveFeaInd)<=numTrain-classNum %���ڿ��н⵫���н��������������ĸ�����ѡ�����еĿ��н������н�����Ĳ����н����ع�
            
            Distances = zeros(length(archiveFeaInd),length(archiveInfeaInd));
            for d1 = 1:length(archiveFeaInd)
                for d2 = 1:length(archiveInfeaInd)
                    for d3 = 1:n
                        Distances(d1,d2) = Distances(d1,d2)+(feaCanX(d1,d3)-infeaCanX(d2,d3))*(feaCanX(d1,d3)-infeaCanX(d2,d3));
                    end
                    Distances(d1,d2) = sqrt(Distances(d1,d2));
                end
            end
            mergeDisVec = max(Distances,[],1);%Ϊ��ֹ����
            [~,sortVecInd] = sort(mergeDisVec);
            numInfea = numTrain - length(archiveFeaInd);
            selInfeaInd = archiveInfeaInd(sortVecInd(1:numInfea));
            conTrainX = [archiveX(archiveFeaInd,:);archiveX(selInfeaInd,:)];
            conTrainC = [archiveC(archiveFeaInd,:);archiveC(selInfeaInd,:)];
        elseif ~isempty(archiveFeaInd) && length(archiveInfeaInd)<=classNum %���ڿ��н⵫�����н��������������ĸ�����ѡ�����еĲ����н���벻���н�����Ŀ��н����ع�
           
            Distances = zeros(length(archiveInfeaInd),length(archiveFeaInd));
            for d1 = 1:length(archiveInfeaInd)
                for d2 = 1:length(archiveFeaInd)
                    for d3 = 1:n
                        Distances(d1,d2) = Distances(d1,d2)+(infeaCanX(d1,d3)-feaCanX(d2,d3))*(infeaCanX(d1,d3)-feaCanX(d2,d3));
                    end
                    Distances(d1,d2) = sqrt(Distances(d1,d2));
                end
            end
            mergeDisVec = max(Distances,[],1);%Ϊ��ֹ����
            [~,sortVecInd] = sort(mergeDisVec);
            numFea = numTrain - length(archiveInfeaInd);
            selFeaInd = archiveFeaInd(sortVecInd(1:numFea));
            conTrainX = [archiveX(archiveInfeaInd,:);archiveX(selFeaInd,:)];
            conTrainC = [archiveC(archiveInfeaInd,:);archiveC(selFeaInd,:)];
        elseif length(archiveFeaInd)>numTrain-classNum && length(archiveInfeaInd)>classNum %���н��벻���н������classNum���������н������classNum��
         
            conTrainFeaX = archiveX(length(archiveFeaInd)-(numTrain-classNum)+1:length(archiveFeaInd),:);
            conTrainFeaC = archiveC(length(archiveFeaInd)-(numTrain-classNum)+1:length(archiveFeaInd),:);
            Distances = zeros(size(conTrainFeaX,1),length(archiveInfeaInd));
            for d1 = 1:size(conTrainFeaX,1)
                for d2 = 1:length(archiveInfeaInd)
                    for d3 = 1:n
                        Distances(d1,d2) = Distances(d1,d2)+(conTrainFeaX(d1,d3)-infeaCanX(d2,d3))*(conTrainFeaX(d1,d3)-infeaCanX(d2,d3));
                    end
                    Distances(d1,d2) = sqrt(Distances(d1,d2));
                end
            end
            mergeDisVec = max(Distances,[],1);%Ϊ��ֹ����
            [~,sortVecInd] = sort(mergeDisVec);
            numInfea = numTrain - size(conTrainFeaX,1);
            selInfeaInd = archiveInfeaInd(sortVecInd(1:numInfea));
            conTrainX = [conTrainFeaX;archiveX(selInfeaInd,:)];
            conTrainC = [conTrainFeaC;archiveC(selInfeaInd,:)];
        end
    else
        conTrainX = archiveX(1:numTrain,:);
        conTrainC = archiveC(1:numTrain,:);
    end
end

