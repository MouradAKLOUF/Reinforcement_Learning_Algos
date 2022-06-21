
%%
clc, clear, close all,

%% epsi-Greedy

nbr_bandits = 10;
nbr_iterations = 1000;
nbr_experiments = 2000;
eps=[1/128, 1/32, 1/16, 1/8, 1/4];

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

%% Upper-Confidence-Bound

avrgRewards_UBC = zeros(nbr_iterations,length(eps));  %for each eps
c=[1/16,1/4,1/2, 2,4];

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

%% Gradient Bandits

alpha=[1/32, 1/8, 1/4, 1, 2];
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


%% Greedy with optimistic initialization
alpha=0.1;
Q0=[1/4, 1/2, 1, 2, 4];
avrgRewards_eps_optimist = zeros(nbr_iterations,length(eps));  %for each eps

for i=1:length(Q0)
    avrg_history_q= zeros (nbr_iterations,1);
    
    for j=1:nbr_experiments
        means_q=randn (1,nbr_bandits);
        
        N_a= zeros (nbr_bandits,1);
        estimated_Q= Q0(i) * ones (nbr_bandits,1);
        history_q= zeros (nbr_iterations,1);
        for k=1:nbr_iterations
            [tempo, chosen_bandit] = max (estimated_Q);
            N_a(chosen_bandit,1) = N_a (chosen_bandit,1)+1;
            reward= means_q(chosen_bandit)+ randn (1,1);
            history_q(k,1)= reward;
            estimated_Q(chosen_bandit,1)=estimated_Q(chosen_bandit,1)+ (alpha)*(reward-estimated_Q(chosen_bandit,1) );
        end
        avrg_history_q= avrg_history_q+ history_q;
    end
    
    avrgRewards_eps_optimist (:,i) = avrg_history_q/nbr_experiments;
end

%% ploting

eps=[1/128, 1/32, 1/16, 1/8, 1/4];
c=[1/16,1/4,1/2, 2,4];
alpha=[1/32, 1/8, 1/4, 1, 2];
Q0=[1/4, 1/2, 1, 2, 4];

avrgRewards_eps_= mean (avrgRewards_eps);
avrgRewards_UBC_= mean (avrgRewards_UBC);
avrgRewards_Gradient_= mean (avrgRewards_Gradient);
avrgRewards_Gradient_nobaseline_= mean (avrgRewards_Gradient_nobaseline);
avrgRewards_eps_optimist_= mean (avrgRewards_eps_optimist);

figure;
semilogx(eps,avrgRewards_eps_ , '-*r');
hold on
semilogx(c, avrgRewards_UBC_, '-*b');
semilogx(alpha,avrgRewards_Gradient_, '-*g');
semilogx(alpha,avrgRewards_Gradient_nobaseline_, '-*c');
semilogx(Q0,avrgRewards_eps_optimist_, '-*k');
hold off
legend('_eps_', '_UBC_','_Gradient_', '_Gradient_NoBaseline_','_greedy_optimistInit_')

xlabel('eps c alpha Q0'); ylabel('avrgRewards');