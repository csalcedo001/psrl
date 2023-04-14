def get_grid_from_file(file_path):
    with open(file_path, 'r') as in_file:
        grid = in_file.readlines()
        grid = [row.splitlines()[0] for row in grid]
    
    cols = len(grid[0])
    for i in range(len(grid)):
        if len(grid[i]) != cols:
            raise ValueError('Invalid grid. Make sure all columns have the same length')
    
    return grid