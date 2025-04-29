import numpy as np
import pandas as pd

def calculate_spring_matrix(params, coordinates):
    """
    params: dict，include d, t, d_b, f_u, t_s, f_y, E, L
    coordinates: np.array, shape=(n,2)
    """
    d = params['d']
    t = params['t']
    d_b = params['d_b']
    f_u = params['f_u']
    t_s = params['t_s']
    f_y = params['f_y']
    E = params['E']
    L = params['L']

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    r_distance = np.sqrt(x ** 2 + y ** 2)
    cos_t = x / r_distance
    sin_t = y / r_distance

    n = coordinates.shape[0]
    Spring_Matrix = np.zeros((48, n))

    fh_0 = 0.082 * (1 - 0.01 * d_b) * d

    def embedment_strength_20(rou, cosine, sine, bolt_d):
        fhi_0 = 0.082 * (1 - 0.01 * bolt_d) * rou
        fhi = fhi_0 / ((1.35 + 0.015 * bolt_d) * np.square(cosine) + np.square(sine))
        return fhi

    def embedment_strength_100(rou, cosine, sine, bolt_d):
        fhi_0 = 0.082 * (1 - 0.01 * bolt_d) * rou * 0.4
        fhi = fhi_0 / ((1.35 + 0.015 * bolt_d) * np.square(cosine) + np.square(sine))
        return fhi

    def Mode_I(r, fh, bolt_d, t):
        return fh * bolt_d * t * r

    def Mode_II(r, fh, bolt_d, t, fu, rou, coef):
        My = 0.3 * fu * coef * np.power(bolt_d, 2.6)
        fa = 20e-6 * rou**2 * bolt_d * t
        return (fh * t * bolt_d * (np.sqrt(2 + (4 * My) / (fh * bolt_d * t**2)) - 1) + fa) * r

    def Mode_III(r, fh, bolt_d, t, fu, rou, coef):
        My = 0.3 * fu * coef * np.power(bolt_d, 2.6)
        fa = 20e-6 * rou**2 * bolt_d * t
        return (2.3 * np.sqrt(My * fh * bolt_d) + fa) * r

    def R_I(fh, rou, bolt_d, fh0, r, Fr):
        kr = (fh / fh0) * (rou / 1.33333)**1.5 * bolt_d / 23 * 2 * r**2
        return Fr / kr

    def R_II(E_coef, bolt_d, L, r, Fr):
        kr = 48 * E * E_coef * np.pi * bolt_d**4 / 64 / L**3 * r**2
        return Fr / kr

    def R_III(t_s, f_y, bolt_d, r, Fr):
        kr = 145 * t_s * f_y * (bolt_d / 25.4)**0.8 * r**2
        return Fr / kr

    fh_20 = np.zeros(n)
    fh_100 = np.zeros(n)

    for i in range(n):
        fh_20[i] = embedment_strength_20(d, cos_t[i], sin_t[i], d_b)
        fh_100[i] = embedment_strength_100(d, cos_t[i], sin_t[i], d_b)

        ## Spring-1-1
        Spring_Matrix[0, i] = Mode_I(r_distance[i], fh_20[i], d_b, t)
        Spring_Matrix[1, i] = R_I(fh_20[i], d, d_b, fh_0, r_distance[i], Spring_Matrix[0, i])
        Spring_Matrix[2, i] = Mode_I(r_distance[i], fh_100[i], d_b, t)
        Spring_Matrix[3, i] = Spring_Matrix[1, i]
        Spring_Matrix[4, i] = Spring_Matrix[0, i]
        Spring_Matrix[5, i] = Spring_Matrix[1, i]
        Spring_Matrix[6, i] = Spring_Matrix[2, i]
        Spring_Matrix[7, i] = Spring_Matrix[1, i]
        Spring_Matrix[8, i] = Spring_Matrix[0, i]
        Spring_Matrix[9, i] = Spring_Matrix[1, i]
        Spring_Matrix[10, i] = Spring_Matrix[2, i]
        Spring_Matrix[11, i] = Spring_Matrix[1, i]
        Spring_Matrix[12, i] = Spring_Matrix[0, i]
        Spring_Matrix[13, i] = Spring_Matrix[1, i]
        Spring_Matrix[14, i] = Spring_Matrix[2, i]
        Spring_Matrix[15, i] = Spring_Matrix[1, i]

        ## Spring-1-2
        Spring_Matrix[16, i] = Mode_II(r_distance[i], fh_20[i], d_b, t, f_u, d, 1)
        Spring_Matrix[17, i] = R_II(1, d_b, L, r_distance[i], Spring_Matrix[16, i])
        Spring_Matrix[18, i] = Mode_II(r_distance[i], fh_100[i], d_b, t, f_u, d, 1)
        Spring_Matrix[19, i] = R_II(1, d_b, L, r_distance[i], Spring_Matrix[18, i])
        Spring_Matrix[20, i] = Spring_Matrix[16, i]
        Spring_Matrix[21, i] = R_II(0.7, d_b, L, r_distance[i], Spring_Matrix[20, i])
        Spring_Matrix[22, i] = Spring_Matrix[18, i]
        Spring_Matrix[23, i] = R_II(0.7, d_b, L, r_distance[i], Spring_Matrix[22, i])
        Spring_Matrix[24, i] = Mode_II(r_distance[i], fh_20[i], d_b, t, f_u, d, 0.5)
        Spring_Matrix[25, i] = R_II(0.6, d_b, L, r_distance[i], Spring_Matrix[24, i])
        Spring_Matrix[26, i] = Mode_II(r_distance[i], fh_100[i], d_b, t, f_u, d, 0.5)
        Spring_Matrix[27, i] = R_II(0.6, d_b, L, r_distance[i], Spring_Matrix[26, i])
        Spring_Matrix[28, i] = Mode_II(r_distance[i], fh_20[i], d_b, t, f_u, d, 0.11)
        Spring_Matrix[29, i] = R_II(0.09, d_b, L, r_distance[i], Spring_Matrix[28, i])
        Spring_Matrix[30, i] = Mode_II(r_distance[i], fh_100[i], d_b, t, f_u, d, 0.11)
        Spring_Matrix[31, i] = R_II(0.09, d_b, L, r_distance[i], Spring_Matrix[30, i])

        ## Spring-1-3
        Spring_Matrix[32, i] = Mode_III(r_distance[i], fh_20[i], d_b, t, f_u, d, 1)
        Spring_Matrix[33, i] = R_III(t_s, f_y, d_b, r_distance[i], Spring_Matrix[32, i])
        Spring_Matrix[34, i] = Mode_III(r_distance[i], fh_100[i], d_b, t, f_u, d, 1)
        Spring_Matrix[35, i] = R_III(t_s, f_y, d_b, r_distance[i], Spring_Matrix[34, i])
        Spring_Matrix[36, i] = Spring_Matrix[32, i]
        Spring_Matrix[37, i] = Spring_Matrix[33, i]
        Spring_Matrix[38, i] = Spring_Matrix[34, i]
        Spring_Matrix[39, i] = Spring_Matrix[35, i]
        Spring_Matrix[40, i] = Mode_III(r_distance[i], fh_20[i], d_b, t, f_u, d, 0.5)
        Spring_Matrix[41, i] = R_III(t_s, f_y, d_b, r_distance[i], Spring_Matrix[40, i])
        Spring_Matrix[42, i] = Mode_III(r_distance[i], fh_100[i], d_b, t, f_u, d, 0.5)
        Spring_Matrix[43, i] = R_III(t_s, f_y, d_b, r_distance[i], Spring_Matrix[42, i])
        Spring_Matrix[44, i] = Mode_III(r_distance[i], fh_20[i], d_b, t, f_u, d, 0.11)
        Spring_Matrix[45, i] = R_III(t_s, f_y, d_b, r_distance[i], Spring_Matrix[44, i])
        Spring_Matrix[46, i] = Mode_III(r_distance[i], fh_100[i], d_b, t, f_u, d, 0.11)
        Spring_Matrix[47, i] = R_III(t_s, f_y, d_b, r_distance[i], Spring_Matrix[46, i])
    Spring_Matrix_swapped = np.transpose(Spring_Matrix, (1, 0))
    return Spring_Matrix_swapped

def save_spring_matrix_as_formatted_csv(Spring_Matrix_swapped, filename='Spring_Matrix.csv'):
    """
    Spring_Matrix_swapped to csv
    """
    timber_temp_pattern = [20, 100, 20, 100, 20, 100, 20, 100]
    steel_temp_pattern = [20, 20, 400, 400, 600, 600, 800, 800]

    def create_table(label1, label2, spring_values):
        df = pd.DataFrame('', index=range(9), columns=[0, 1, 2, 3])  
        df.iloc[0, 0] = label1
        df.iloc[0, 1] = label2
        df.iloc[0, 2] = 'Timber temperature'
        df.iloc[0, 3] = 'Steel temperature'
        for idx in range(8):
            df.iloc[idx+1, 0] = spring_values[2*idx]
            df.iloc[idx+1, 1] = spring_values[2*idx+1]
            df.iloc[idx+1, 2] = timber_temp_pattern[idx]
            df.iloc[idx+1, 3] = steel_temp_pattern[idx]
        return df

    final_rows = []

    for i in range(Spring_Matrix_swapped.shape[0]):
        data = Spring_Matrix_swapped[i, :]  # 48列

        spring1_1 = data[0:16]
        spring1_2 = data[16:32]
        spring1_3 = data[32:48]

        df1 = create_table(f'spring{i+1}-1 Moment (N·mm)', f'spring{i+1}-1 Rotation (rad)', spring1_1)
        df2 = create_table(f'spring{i+1}-2 Moment (N·mm)', f'spring{i+1}-2 Rotation (rad)', spring1_2)
        df3 = create_table(f'spring{i+1}-3 Moment (N·mm)', f'spring{i+1}-3 Rotation (rad)', spring1_3)

        combined = pd.concat([df1, df2, df3], axis=1)

        final_rows.append(combined)

    final_result = pd.concat(final_rows, axis=0, ignore_index=True)

    final_result.to_csv(filename, index=False, header=False, encoding='utf-8-sig')
