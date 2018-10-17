function compute_effective_squash( )

% This function is supposed to show what the single, self-exciting unit
% converges to. If you change the last parameter, you get qualitatively
% different final output values for a certain range of external input_level
% values. 

itr = 0; 

for external_input = -1 : 0.01 : 1
    itr = itr + 1;
    output(itr) =self_excite( external_input, 4, 0.5, 1, 0.9 );
    
end

%figure
plot( [-1:0.01:1], output, 'r' ) % The letter indicates color
xlabel('External input level')
ylabel('Output level (V)')

