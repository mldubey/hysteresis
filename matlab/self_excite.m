function  output = self_excite( external_input, L, B, self_wt, initial_V)
% This function tries to compute the final, equilibrium output value for a
% self-exciting unit with a particular external input level, and a
% particular initial output value (initial_V).

% V is the unit's output:
V(1) = initial_V;

dt = 0.01;

for i = 1:1000
    internal_input = self_wt * V(i);
    input = external_input + internal_input;
    
    dV = -V(i) + 1./(1+exp(-L.*(input - B)));

    V(i+1) = V(i) + dV*dt;
end

output = V(end);

% figure
% plot(V)
% ylim([0 1])

% keyboard

end

