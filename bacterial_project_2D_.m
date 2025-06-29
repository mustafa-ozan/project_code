% project 2D Version of single bacterium motion in one stable transmitter
% one stable receiver
%
%
% programmer: MUSTAFA OZAN DUMAN
%
% 29.06.2025

clear all;
clc;

nanomachine.radius = 50;%um * 10^-6;

nanomachine.number = 2;
nanomachine.infos = zeros(nanomachine.number, 2); % X Y coords
nanomachine.infos(1, :) = [100 100];
nanomachine.distance_btw_NMs = 1000;
nanomachine.infos(2, :) = [100+nanomachine.distance_btw_NMs 100];

nanomachine.attractant_releasing_rate = 10^-15; % mol/s
nanomachine.attractant_diffusion_coefficient = 10^3; %um2/s at water

%nanomachine.attractant_concentration = Q / (4 * D * pi * r);
%serine threshold 3*10^-7 M ==> M = mol / cm^3
nanomachine.serine_chemotaxis_threshold = 3 * 10^-7 * 10^-15; % mol/um^3
nanomachine.source_range = nanomachine.attractant_releasing_rate / (4 * pi * nanomachine.serine_chemotaxis_threshold * nanomachine.attractant_diffusion_coefficient);

bacteria.length = 2;%um * 10^-6;
bacteria.wide = 1;%um * 10^-6;
bacteria.radius = 1;%um 
bacteria.tumble_angle_while_gradient = 68;
bacteria.tumble_length = 0.1;

%%% run speed will be determined completely from figure %%%
bacteria.expected_run_speed_no_attractant = 14.2; % um/s for no attractant enviorenment => canceled
bacteria.expected_run_time_at_no_attractant = 0.86; %second expected wrt berg 1972 +-1.18
bacteria.run_time_min = 0.86; %second +- 1.18
bacteria.lifespan = 15; % minutes to die
bacteria.error_count = 0; % in case of error

%bacteria info holder matrix
bacteria.infos = zeros(1, 9);

env_params.length = 10^6; % um10^-2;
env_params.wide = 10^6; % um10^-2;
env_params.D_inMicroMeterSqrPerSecond = 79.4;

sim_params.delta_t = 0.1;%second
sim_params.trial_number = 100;

global bacteria_reach_time_all_simulations_holder; % it holds bacteria reach_time
bacteria_reach_time_all_simulations_holder = zeros(0, 1); %initialized

global bacteria_success_percent_all_simulations_holder; % it holds bacteria reach_time
bacteria_success_percent_all_simulations_holder = zeros(0, 1); %initialized


for lifespan = 5 : 5 : 30

    bacteria.lifespan = lifespan;

    for attractant_releasing_rate = 1 : 3
    
        nanomachine.attractant_releasing_rate = 10^-15 * 10 ^ ( attractant_releasing_rate - 1 );
    
        for distance_btw_NMs = 150 : 50 : 13500
            nanomachine.distance_btw_NMs = distance_btw_NMs;
            nanomachine.infos(2, :) = [100+nanomachine.distance_btw_NMs 100];
            % nanomachine.infos(2, :) = [431 100];
            [success_number, average_reach_time] = simplified_simulation( nanomachine, bacteria, sim_params );
        
        end
    end
end
writematrix( bacteria_reach_time_all_simulations_holder, 'bacteria_2D_results_01.05.2025.xlsx' );

beep;


function [success_number, average_reach_time] = simplified_simulation(nanomachine, bacteria, sim_params)

    global bacteria_reach_time_all_simulations_holder; % it holds bacteria reach_time
    global bacteria_success_percent_all_simulations_holder;

    reach_time = zeros(1, sim_params.trial_number );

    for trial = 1 : sim_params.trial_number
    
        %%%%%%%%bacterium initializition

        bacteria.infos(1, 1) = 152;
        bacteria.infos(1, 2) = 102;
        bacteria.infos(1, 3) = 152;
        bacteria.infos(1, 4) = 102;
    
        bacteria.infos(1, 5) = 1; % +x direction
        bacteria.infos(1, 6) = 0;
    
        bacteria.infos(1, 9) = bacteria.lifespan * 60;  % death clock initialized seconds
    
        % run speed calculator run time calculator    
    
        bacterium_to_receiver_vector  = ( nanomachine.infos( 2, : ) - bacteria.infos( 1 , 3:4 ) );
    
        bacterium_to_receiver_current_position_distance  = sqrt( sum( bacterium_to_receiver_vector.^2, 2) );
            
        bacterium_to_receiver_vector  = bacterium_to_receiver_vector / bacterium_to_receiver_current_position_distance;
            
        bacterium_concentration_current = nanomachine.attractant_releasing_rate / ( 4 * pi * nanomachine.attractant_diffusion_coefficient * bacterium_to_receiver_current_position_distance ) ;
            
        cos_of_angle  = sum( bacterium_to_receiver_vector .* bacteria.infos( 1 , 5:6 ), "all" ); % their magnitudes are 1, so divide with 1

        angle_in_degree = acosd( cos_of_angle );

        x = log10( bacterium_concentration_current * 10^15 ); %um3 to cm3

        fractional_change_run_time_at_current_distance = 0.02375 * x^5 + 0.49097 * x^4 + 3.79279 * x^3 + 13.35690 * x^2 + 21.24521 * x + 15.59139;

        normal_value_run_time_at_current_distance = fractional_change_run_time_at_current_distance * bacteria.expected_run_time_at_no_attractant;

        if bacterium_concentration_current > nanomachine.serine_chemotaxis_threshold % in threshold
        
            if angle_in_degree <= 90 % in gradient
    
                % the run speed calculated 
                bacteria.infos( 1, 7 ) = ( bacteria.expected_run_speed_no_attractant * 1.4 * ( 1.02 - 0.02 * angle_in_degree / 90 ) );                            
                % run time calculated the same technique
                bacteria.infos( 1, 8 ) = ( round( normal_value_run_time_at_current_distance * ( 1.48 - 0.48 * angle_in_degree / 90 ), 1 ) );
                        
            else %if angle_in_degree > 90 % not in gradient
    
                % the run speed calculated
                bacteria.infos( 1, 7 ) = bacteria.expected_run_speed_no_attractant * 1.4;
                % run time calculated the same technique
                bacteria.infos( 1, 8 ) = round( normal_value_run_time_at_current_distance, 1 );
                
            % else    % error
            %     bacteria.error_count = bacteria.error_count + 1;
            end

        else % not enough attractant
            bacteria.infos( 1, 7 ) = bacteria.expected_run_speed_no_attractant;
            bacteria.infos( 1, 8 ) = round( bacteria.expected_run_time_at_no_attractant, 1);
        end         
        %%%%%%%%%%%%%%%% bacterium initialized
        
        
        %%%%%%%%%%%%%%%% simulation is initialized
    
        for t = 0 : sim_params.delta_t : ( 60 * bacteria.lifespan + 1 ) % minutes of lifespan
    
            if bacteria.infos(1, 8) > 0.005 % in run period
    
                bacteria.infos(1, 1:2) = bacteria.infos(1, 3:4); % prev_cor = curr_cor
    
                bacteria.infos(1, 3:4) = bacteria.infos(1, 3:4) + bacteria.infos(1, 5:6) * sim_params.delta_t * bacteria.infos(1, 7); % curr_cor updated
                
                bacteria.infos(1, 8) = bacteria.infos(1, 8) - sim_params.delta_t; % run time decreased
    
    
                bacterium_to_receiver_vector  = ( nanomachine.infos( 2, : ) - bacteria.infos( 1 , 3:4 ) );
            
                bacterium_to_receiver_current_position_distance  = sqrt( sum( bacterium_to_receiver_vector.^2, "all") );
            
            
                bacterium_to_source_vector  = ( nanomachine.infos( 1, : ) - bacteria.infos( 1 , 3:4 ) );
            
                bacterium_to_source_current_position_distance  = sqrt( sum( bacterium_to_source_vector.^2, "all") );
            
    
    
                if bacterium_to_receiver_current_position_distance <= ( nanomachine.radius + 1 ) % destination reach

                    % plot([bacteria.infos(1, 1) bacteria.infos(1, 3)], [bacteria.infos(1, 2) bacteria.infos(1, 4)], 'k'); %plot
                    % termination_time = t;
                    reach_time (1, trial) = t;
                    break; % terminate program

                elseif bacterium_to_source_current_position_distance < ( nanomachine.radius + 1 ) % source collision

                    bacteria.infos( 1, 3:4 ) = bacteria.infos( 1, 1:2 ); % return to the previous coordinates
                    bacteria.infos( 1, 8 ) = 0; % go to tumble period

                else

                end

                bacteria.infos(1, 9) = bacteria.infos(1, 9) - sim_params.delta_t; % death time updated
                if bacteria.infos(1, 9) < 0.005 % death time reached
                    
                    % termination_time = t;
                    % bacterium_death_time_reached = 1;
                    reach_time(1, trial) = 0;
                    break;
                end

            else % in tumble period
    
                while(1)
                    random_tumble_angle = normrnd(68,36); % * ((rand(1,1) > 0.5)*2 - 1);
                    if 0 < random_tumble_angle && random_tumble_angle < 180 %desired range
                        break;
                    end
                end

                bacterium_to_receiver_vector  = nanomachine.infos( 2, : ) - bacteria.infos( 1 , 3:4 );
                
                bacterium_to_receiver_current_position_distance  = sqrt( sum( bacterium_to_receiver_vector.^2, "all" ) );
                
                bacterium_to_receiver_vector  = bacterium_to_receiver_vector / bacterium_to_receiver_current_position_distance;
                
                bacterium_concentration_current = nanomachine.attractant_releasing_rate / ( 4 * pi * nanomachine.attractant_diffusion_coefficient * bacterium_to_receiver_current_position_distance ) ;

                %%%%
                % current dir angle between x axes
                bacterium_curr_dir_angle_in_degree = atan2d(  bacteria.infos( 1 , 3 ), bacteria.infos( 1 , 4 ) );
                
                if bacterium_curr_dir_angle_in_degree < 0
                    bacterium_curr_dir_angle_in_degree = bacterium_curr_dir_angle_in_degree + 360;
                end
                
                % current dir angle between x axes
                bacterium_to_receiver_curr_angle_in_degree = atan2d(  bacterium_to_receiver_vector( 1 , 1 ), bacterium_to_receiver_vector( 1 , 2 ) );
                
                if bacterium_to_receiver_curr_angle_in_degree < 0
                    bacterium_to_receiver_curr_angle_in_degree = bacterium_to_receiver_curr_angle_in_degree + 360;
                end
                
                new_angle_one = bacterium_curr_dir_angle_in_degree + random_tumble_angle;
                if new_angle_one < 0
                    new_angle_one = new_angle_one + 360;
                elseif new_angle_one > 360
                    new_angle_one = new_angle_one - 360;
                else
                
                end
                
                new_angle_two = bacterium_curr_dir_angle_in_degree - random_tumble_angle;
                if new_angle_two < 0
                    new_angle_two = new_angle_two + 360;
                elseif new_angle_two > 360
                    new_angle_two = new_angle_two - 360;
                else
                    
                end

                %%%%%%%%%%% if it is in range, assign better angle, else randomly assign
                if bacterium_concentration_current > nanomachine.serine_chemotaxis_threshold % in threshold
                    difference_btw_angles_one = abs( new_angle_one - bacterium_to_receiver_curr_angle_in_degree );
                    if difference_btw_angles_one > 180
                        difference_btw_angles_one = 360 - difference_btw_angles_one;
                    end
                    
                    difference_btw_angles_two = abs( new_angle_two - bacterium_to_receiver_curr_angle_in_degree );
                    if difference_btw_angles_two > 180
                        difference_btw_angles_two = 360 - difference_btw_angles_two;
                    end
                    
                    if difference_btw_angles_one > difference_btw_angles_two % second is better
                        bacteria.infos( 1 , 5:6 ) = [ sind(new_angle_two) cosd(new_angle_two) ];
                    else % first is better
                        bacteria.infos( 1 , 5:6 ) = [ sind(new_angle_one) cosd(new_angle_one) ];
                    end
                else % not in range, random angle assignment
                    if ((rand(1,1) > 0.5)*2 - 1) == 1 % randomly select one
                        bacteria.infos( 1 , 5:6 ) = [ sind(new_angle_one) cosd(new_angle_one) ];
                    else
                        bacteria.infos( 1 , 5:6 ) = [ sind(new_angle_two) cosd(new_angle_two) ];
                    end
                end

                                
                % bacterium_to_receiver_gradient  = nanomachine.attractant_releasing_rate / ( 4 * pi * nanomachine.attractant_diffusion_coefficient * bacterium_to_receiver_current_position_distance ^ 2 );
                                
                cos_of_angle  = sum( bacterium_to_receiver_vector .* bacteria.infos( 1 , 5:6 ), "all" ); % their magnitudes are 1, so divide with 1
                
                angle_in_degree = acosd( cos_of_angle );
                    
                x = log10( bacterium_concentration_current * 10^15 ); %um3 to cm3
                    
                fractional_change_run_time_at_current_distance = 0.02375 * x^5 + 0.49097 * x^4 + 3.79279 * x^3 + 13.35690 * x^2 + 21.24521 * x + 15.59139;
                    
                normal_value_run_time_at_current_distance = fractional_change_run_time_at_current_distance * bacteria.expected_run_time_at_no_attractant;
                
                if bacterium_concentration_current > nanomachine.serine_chemotaxis_threshold % in threshold
                
                    if angle_in_degree <= 90 % in gradient
                
                        % the run speed calculated 
                        bacteria.infos( 1, 7 ) = ( bacteria.expected_run_speed_no_attractant * 1.4 * ( 1.02 - 0.02 * angle_in_degree / 90 ) );                            
                        
                        % run time calculated the same technique
                        bacteria.infos( 1, 8 ) = ( round( normal_value_run_time_at_current_distance * ( 1.48 - 0.48 * angle_in_degree / 90 ), 1 ) );

                        bacteria.infos( 1, 8 ) = round( bacteria.infos( 1, 8 ) , 1 ); % correction after all

                    elseif angle_in_degree > 90 % not in gradient

                        % the run speed calculated
                        bacteria.infos( 1, 7 ) = bacteria.expected_run_speed_no_attractant * 1.4;

                        % run time calculated the same technique
                        bacteria.infos( 1, 8 ) = round( normal_value_run_time_at_current_distance, 1 );

                        bacteria.infos( 1, 8 ) = round( bacteria.infos( 1, 8 ) , 1 ); % correction after all

                    else    % error
                        bacteria.error_count = bacteria.error_count + 1;
                    end

                else % not enough attractant

                    bacteria.infos( 1, 7 ) = bacteria.expected_run_speed_no_attractant;
                    bacteria.infos( 1, 8 ) = round( bacteria.expected_run_time_at_no_attractant, 1); 
                end
                    
                bacteria.infos(1, 9) = bacteria.infos(1, 9) - sim_params.delta_t; % death time updated
                if bacteria.infos(1, 9) < 0.005 % death time reached
                    
                    % termination_time = t;
                    % bacterium_death_time_reached = 1;
                    reach_time(1, trial) = 0;
                    break;
                end
            end
        end
    end

    success_number = nnz( reach_time ) ;

    average_reach_time = sum( reach_time, "all") / success_number;

    success_percent = 100 * ( success_number / sim_params.trial_number );

    bacteria_reach_time_all_simulations_holder = [ bacteria_reach_time_all_simulations_holder; log10(nanomachine.attractant_releasing_rate) nanomachine.distance_btw_NMs bacteria.lifespan average_reach_time  success_percent ];

end
