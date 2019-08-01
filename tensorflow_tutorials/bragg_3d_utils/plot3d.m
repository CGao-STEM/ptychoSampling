img_recons = importdata('cell_recons_var.mat');
img_true = importdata('cell_true_var.mat');
%img_recons = permute(img_recons, [2, 3, 1]);
%img_true = permute(img_true, [2, 3, 1]);
img_true = permute(img_true, [2, 1, 3]);
img_recons = permute(img_recons, [2, 1, 3]);
img_true = img_true(9:69, :, 9:73);
img_recons = img_recons(9:69, :, 9:73); 
[ x, y, z ] = meshgrid( 0:size( img_recons, 2 )-1, 0:size( img_recons, 1 )-1, 0:size( img_recons, 3 )-1 );
imgs = {img_true, img_recons};
img_names = ["cell_true", "cell_recons"];

for i = 1:2
    % subplot( 1, 2, 1 );
    img = imgs{i};
    %img = img * exp(1j * pi);
    %angle_unwrapped = unwrap(unwrap(unwrap(angle(img),[], 1), [], 2), [],3);
    %angle_unwrapped = unwrap(angle(img), [], 3)
    %angles_smoothed = smooth3(angle_unwrapped, 'gaussian', 3);
    %angles_wrapped = angle(exp(1j * angles_smoothed));
    
    figure; 
    
    quiver3([0 0 0], [0 0 0], [0 0 0],...
        [10 0 0], [0 10 0], [0 0 10],...
        'linewidth', 3, 'color', [0.4 0.1 0.8],...
        'maxheadsize', 0.7); hold on;
    text( [12 0 0], [0 12 0], [0 0 12],...
        {'x' 'y' 'z'}, 'FontSize',25);
    
    isosurface( ...
        x, y, z, ...
        smooth3( abs( img ), 'gaussian', 9), 0.4, ...
        smooth3( angle( img), 'gaussian', 9 ) ); 
    
    c = colorbar; 
    caxis([-pi pi]);
    c.Ticks = [-pi  0  pi ];
    c.TickLabels ={'-\pi', 0, '\pi'};
    c.FontSize = 25;
    c.AxisLocation = 'out';
    c.Location = 'east';
    c_position = c.Position;
    c.Position = [c_position(1) c_position(2) c_position(3) c_position(4)*0.9]    
    
    axis square; 
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset; 
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width * 1.05 ax_height];
    
    
    axis('off', 'image');
    view(69, 26);
    lighting gouraud;  
    camlight('right'); 
    
    fig = gcf;
    fig_orig_pos = fig.Position;
    fig.Position = [fig_orig_pos(1) fig_orig_pos(2) fig_orig_pos(3)*0.9 fig_orig_pos(4)];
    fig.PaperPositionMode = 'auto'
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
    
    saveas(fig, img_names(i), 'pdf');

end