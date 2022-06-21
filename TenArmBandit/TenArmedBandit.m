clc, clear, close all,
%%

nbr_bandits = 10;
mean_q = randn(nbr_bandits,1);
sample_size = 1000;
action_rewards= zeros (nbr_bandits, sample_size ) ;

for a=1:nbr_bandits
    possible_rewards = mean_q(a)+ randn (sample_size,1);
    action_rewards (a,: ) = possible_rewards;
end
                                                      
figure;
for i=1:10
subplot(5,2,i),histogram(action_rewards (i,: ),255)
end

%%
clc, clear, close all,

%% epsi-Greedy

nbr_bandits = 10;
nbr_iterations = 1000;
nbr_experiments = 2000;
eps=[0.1, 0.01, 0];

avrgRewards_eps = zeros(nbr_iterations,length(eps));  %for each eps

for i=1:length(eps)
    avrg_history_q= zeros (nbr_iterations,1);
    
    for j=1:nbr_experiments
        means_q=randn (1,nbr_bandits);
        
        N_a= zeros (nbr_bandits,1);
        estimated_Q= zeros (nbr_bandits,1);
        history_q= zeros (nbr_iterations,1);
        for k=1:nbr_iterations
            if rand(1,1) > eps(i)
                [tempo, chosen_bandit] = max (estimated_Q);
            else
                chosen_bandit= randi([1 10],1);
            end
            N_a(chosen_bandit,1) = N_a (chosen_bandit,1)+1;
            reward= means_q(chosen_bandit)+ randn (1,1);
            history_q(k,1)= reward;
            estimated_Q(chosen_bandit,1)=estimated_Q(chosen_bandit,1)+ (1/N_a (chosen_bandit,1))*(reward-estimated_Q(chosen_bandit,1) );
        end
        avrg_history_q= avrg_history_q+ history_q;
    end
    
    avrgRewards_eps (:,i) = avrg_history_q/nbr_experiments;
end

%% Cst LR:  constant learning rate 

avrgRewards_eps_cstLR = zeros(nbr_iterations,length(eps));  %for each eps
LR=0.1;

for i=1:length(eps)
    avrg_history_q= zeros (nbr_iterations,1);
    
    for j=1:nbr_experiments
        means_q=randn (1,nbr_bandits);
        
        N_a= zeros (nbr_bandits,1);
        estimated_Q= zeros (nbr_bandits,1);
        history_q= zeros (nbr_iterations,1);
        for k=1:nbr_iterations
            if rand(1,1) > eps(i)
                [tempo, chosen_bandit] = max (estimated_Q);
            else
                chosen_bandit= randi([1 10],1);
            end
            N_a(chosen_bandit,1) = N_a (chosen_bandit,1)+1;
            reward= means_q(chosen_bandit)+ randn (1,1);
            history_q(k,1)= reward;
            estimated_Q(chosen_bandit,1)=estimated_Q(chosen_bandit,1)+ (LR)*(reward-estimated_Q(chosen_bandit,1) );
        end
        avrg_history_q= avrg_history_q+ history_q;
    end
    
    avrgRewards_eps_cstLR (:,i) = avrg_history_q/nbr_experiments;
end


figure;
plot(avrgRewards_eps(:,1), 'g');
hold on
plot(avrgRewards_eps(:,2), 'b');
plot(avrgRewards_eps(:,3), 'r');
plot(avrgRewards_eps_cstLR(:,1), '-m');
plot(avrgRewards_eps_cstLR(:,2), '-c');
plot(avrgRewards_eps_cstLR(:,3), '-y');
hold off
legend('eps=0.1', 'eps=0.01', 'eps=0','cst LR=0.1 & eps=0.1', 'cst LR=0.1 & eps=0.01', 'cst LR=0.1 & eps=0')

%% Upper-Confidence-Bound

avrgRewards_UBC = zeros(nbr_iterations,length(eps));  %for each eps
c=[ 2,4,6];

for i=1:length(c)
    avrg_history_q= zeros (nbr_iterations,1);
    
    for j=1:nbr_experiments
        means_q=randn (1,nbr_bandits);
        
        N_a= zeros (nbr_bandits,1);
        estimated_Q= zeros (nbr_bandits,1);
        history_q= zeros (nbr_iterations,1);
        for k=1:nbr_iterations 
            [tempo, chosen_bandit] = max ( estimated_Q + c(i)*sqrt(log(k)./ (N_a+0.001))  );
            N_a(chosen_bandit,1) = N_a (chosen_bandit,1)+1;
            reward= means_q(chosen_bandit)+ randn (1,1);
            history_q(k,1)= reward;
            estimated_Q(chosen_bandit,1)=estimated_Q(chosen_bandit,1)+ (1/N_a (chosen_bandit,1))*(reward-estimated_Q(chosen_bandit,1) );
        end
        avrg_history_q= avrg_history_q+ history_q;
    end
    
    avrgRewards_UBC (:,i) = avrg_history_q/nbr_experiments;
end


figure;
plot(avrgRewards_eps(:,1), 'g');
hold on
plot(avrgRewards_eps(:,2), 'b');
plot(avrgRewards_eps(:,3), 'r');
plot(avrgRewards_UBC(:,1), '-m');
plot(avrgRewards_UBC(:,2), '-c');
plot(avrgRewards_UBC(:,3), '-y');
hold off
legend('eps=0.1', 'eps=0.01', 'eps=0','UBC c=2', 'UBC c=4', 'UBC c=6')

figure;
plot(avrgRewards_eps_cstLR(:,1), 'g');
hold on
plot(avrgRewards_eps_cstLR(:,2), 'b');
plot(avrgRewards_eps_cstLR(:,3), 'r');
plot(avrgRewards_UBC(:,1), '-m');
plot(avrgRewards_UBC(:,2), '-c');
plot(avrgRewards_UBC(:,3), '-y');
hold off
legend('cst LR=0.1 & eps=0.1', 'cst LR=0.1 & eps=0.01', 'cst LR=0.1 & eps=0','UBC c=2', 'UBC c=4', 'UBC c=6')

%%

%% epsi-Greedy with Optimistic Init

Q0=[1/2, 2, 4];
avrgRewards_eps_OptimistInit = zeros(nbr_iterations,length(eps));  %for each eps

for i=1:length(Q0)
    avrg_history_q= zeros (nbr_iterations,1);
    
    for j=1:nbr_experiments
        means_q=randn (1,nbr_bandits);
        
        N_a= zeros (nbr_bandits,1);
        estimated_Q= Q0(i)*ones (nbr_bandits,1);
        history_q= zeros (nbr_iterations,1);
        for k=1:nbr_iterations
            if rand(1,1) > 0.1
                [tempo, chosen_bandit] = max (estimated_Q);
            else
                chosen_bandit= randi([1 10],1);
            end
            N_a(chosen_bandit,1) = N_a (chosen_bandit,1)+1;
            reward= means_q(chosen_bandit)+ randn (1,1);
            history_q(k,1)= reward;
            estimated_Q(chosen_bandit,1)=estimated_Q(chosen_bandit,1)+ (1/N_a (chosen_bandit,1))*(reward-estimated_Q(chosen_bandit,1) );
        end
        avrg_history_q= avrg_history_q+ history_q;
    end
    
    avrgRewards_eps_OptimistInit  (:,i) = avrg_history_q/nbr_experiments;
end

figure;
plot(avrgRewards_eps(:,1), 'g');
hold on
plot(avrgRewards_eps(:,2), 'b');
plot(avrgRewards_eps(:,3), 'r');
plot(avrgRewards_eps_OptimistInit(:,1), '-m');
plot(avrgRewards_eps_OptimistInit(:,2), '-c');
plot(avrgRewards_eps_OptimistInit(:,3), '-y');
hold off
legend('eps=0.1', 'eps=0.01', 'eps=0','eps=0.1 OptimistInit Q0=2', 'eps=0.1 OptimistInit Q0=4', 'eps=0.1 OptimistInit Q0=6')



%% Gradient Bandits

alpha=[0.1 0.4];
shift=0;

avrgRewards_Gradient= zeros (nbr_iterations, length(alpha)) ;
avrgRewards_Gradient_nobaseline= zeros(nbr_iterations, length(alpha)) ;
avrgOptActionSelections=zeros (nbr_iterations, length(alpha)) ;
avrgOptActionSelections_nobaseline=zeros (nbr_iterations, length(alpha)) ;
 
for i=1:length(alpha)
    avrg_history_q= zeros (nbr_iterations,1);
    avrg_history_q_nobaseline= zeros (nbr_iterations,1);

    OptActionSelections= zeros (nbr_iterations,1);
    OptActionSelections_nobaseline= zeros (nbr_iterations,1);
    
    for j=1:nbr_experiments
        means_q=randn (1,nbr_bandits)+shift;
        
        avgReward=0;
        
        history_q= zeros (nbr_iterations,1);
        history_q_nobaseline= zeros (nbr_iterations,1);
        
        historyOptAction= zeros (nbr_iterations,1);
        historyOptAction_nobaseline= zeros (nbr_iterations,1);
        
        H_a= zeros (nbr_bandits,1);                     % try it with zeros also
        H_a_nobaseline= zeros (nbr_bandits,1);
        
        N_a= zeros (nbr_bandits,1);
        N_a_nobaseline= zeros (nbr_bandits,1);

        for k=1:nbr_iterations 
            %%
            maxi= max(H_a);
            pi_a=exp(H_a-maxi)/sum(exp(H_a-maxi));
            chosen_bandit= getRandomChoice (pi_a);
            N_a(chosen_bandit,1) = N_a (chosen_bandit,1)+1;
            reward= means_q(chosen_bandit)+ randn (1,1);
            avgReward = avgReward + (1 / k) * (reward - avgReward);
            history_q(k,1)= reward;
            for l=1:nbr_bandits 
                if l == chosen_bandit
                    H_a (l,1) = H_a (l,1)+ alpha(i) *(reward-avgReward)*(1-pi_a(l));
                else
                    H_a (l,1) = H_a (l,1)+ alpha(i) *(reward-avgReward)*(pi_a(l));
                end
            end
            [~, maxIdx] = max ( means_q);
            if chosen_bandit == maxIdx
                historyOptAction(k)= 1;
            end
            
            %% no baseline >> avgReward =0
            maxi= max(H_a_nobaseline);
            pi_a_nobaseline=exp(H_a_nobaseline-maxi)/sum(exp(H_a_nobaseline-maxi));
            chosen_bandit_nobaseline= getRandomChoice (pi_a_nobaseline);
            N_a_nobaseline(chosen_bandit_nobaseline,1) = N_a_nobaseline(chosen_bandit_nobaseline,1)+1;
            reward_nobaseline= means_q(chosen_bandit_nobaseline)+ randn (1,1);
            history_q_nobaseline(k,1)= reward_nobaseline;
            for l=1:nbr_bandits 
                if l == chosen_bandit_nobaseline
                    H_a_nobaseline (l,1) = H_a_nobaseline (l,1)+ alpha(i) *(reward_nobaseline)*(1-pi_a_nobaseline(l));
                else
                    H_a_nobaseline (l,1) = H_a_nobaseline (l,1)+ alpha(i) *(reward_nobaseline)*(pi_a_nobaseline(l));
                end
            end
            [~, maxIdx] = max ( means_q);
            if chosen_bandit_nobaseline == maxIdx
                historyOptAction_nobaseline(k)= 1;
            end
            
        end
        %%
        avrg_history_q= avrg_history_q+ history_q;
        avrg_history_q_nobaseline= avrg_history_q_nobaseline+ history_q_nobaseline;
        OptActionSelections=OptActionSelections+historyOptAction;
        OptActionSelections_nobaseline=OptActionSelections_nobaseline+historyOptAction_nobaseline;
        
    end
    
    avrgOptActionSelections(:, i)= OptActionSelections/nbr_experiments;
    avrgOptActionSelections_nobaseline(:, i)= OptActionSelections_nobaseline/nbr_experiments;
    avrgRewards_Gradient (:,i) = avrg_history_q/nbr_experiments;
    avrgRewards_Gradient_nobaseline(:,i) = avrg_history_q_nobaseline/nbr_experiments;
    
end

figure;
plot(avrgRewards_Gradient(:,1), 'r');
hold on
plot(avrgRewards_Gradient(:,2), 'b');
plot(avrgRewards_Gradient_nobaseline(:,1), '-m');
plot(avrgRewards_Gradient_nobaseline(:,2), '-c');
hold off
legend('Gradient alpha=0.1', 'Gradient alpha=0.4','Gradient nobaseline alpha=0.1', 'Gradient nobaseline alpha=0.4')


figure;
plot(avrgOptActionSelections(:,1), 'r');
hold on
plot(avrgOptActionSelections(:,2), 'b');
plot(avrgOptActionSelections_nobaseline(:,1), '-m');
plot(avrgOptActionSelections_nobaseline(:,2), '-c');
hold off
legend('optAction% alpha=0.1', 'optAction% alpha=0.4','optAction% nobaseline alpha=0.1', 'optAction% nobaseline alpha=0.4')

%%

figure;
plot(avrgRewards_Gradient(:,1), 'r');
hold on
plot(avrgRewards_Gradient(:,2), 'b');
plot(avrgRewards_Gradient_nobaseline(:,1), '-m');
plot(avrgRewards_Gradient_nobaseline(:,2), '-c');

plot(avrgRewards_UBC(:,1), '-k');
plot(avrgRewards_UBC(:,2), '-g');
plot(avrgRewards_UBC(:,3), '-y');

hold off
legend('Gradient alpha=0.1', 'Gradient alpha=0.4','Gradient nobaseline alpha=0.1', 'Gradient nobaseline alpha=0.4','UBC c=2', 'UBC c=4', 'UBC c=6')



