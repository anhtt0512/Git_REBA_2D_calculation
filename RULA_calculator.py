import numpy as np

class GetRULAScores :

    def getUpperArmRULA(upper_arm_angle: float , raised : int, abducted : int , supported_leaning : int):
        angle_limits = [[-20, 20], [-90, -20], [20, 45],[45, 90], [90 , 180]]
        upper_arm_score = raised + abducted + supported_leaning

        # scoring for any side
        if upper_arm_angle > angle_limits[0][0] and upper_arm_angle <= angle_limits[0][1]:
           upper_arm_score += 1
        elif upper_arm_angle > angle_limits[1][0] and upper_arm_angle <= angle_limits[1][1]:
           upper_arm_score += 2
        elif upper_arm_angle > angle_limits[2][0] and upper_arm_angle <= angle_limits[2][1]:
            upper_arm_score += 2
        elif upper_arm_angle > angle_limits[3][0] and upper_arm_angle <= angle_limits[3][1]:
            upper_arm_score += 3
        elif upper_arm_angle > angle_limits[4][0] and upper_arm_angle <= angle_limits[4][1]:
            upper_arm_score += 4

        return upper_arm_score

    def getLowerArmRULA(lower_arm_angle : float , midline_side_arm : int ):
        angle_limits = [[60, 100], [0, 60] , [100 , 180]]
        lower_arm_score = midline_side_arm

        # scoring for any side
        if lower_arm_angle > angle_limits[0][0] and lower_arm_angle < angle_limits[0][1]:
            lower_arm_score += 1
        elif lower_arm_angle > angle_limits[1][0] and lower_arm_angle <= angle_limits[1][1]:
            lower_arm_score += 2
        elif lower_arm_angle > angle_limits[2][0] and lower_arm_angle <= angle_limits[2][1]:
            lower_arm_score += 2

        return lower_arm_score
    
    def getWristRULA(wrist_angle : float ,wrist_bent : int, wrist_twisted_or_end_range : int) :
        angle_limits = [[-15, -1], [-1, 1],  [1, 15] , [-90, -15] , [15, 90]]
        wrist_score =  wrist_bent + wrist_twisted_or_end_range

        #scoring for any side
        if wrist_angle > angle_limits[0][0] and wrist_angle < angle_limits[0][1]:
            wrist_score += 2
        elif wrist_angle > angle_limits[1][0] and wrist_angle <= angle_limits[1][1]:
            wrist_score += 1
        elif wrist_angle >= angle_limits[2][0] and wrist_angle < angle_limits[2][1]:
            wrist_score += 2
        elif wrist_angle >= angle_limits[3][0] and wrist_angle < angle_limits[3][1]:
            wrist_score += 3
        elif wrist_angle > angle_limits[3][0] or wrist_angle <= angle_limits[3][1]:
            wrist_score += 3
        
        return wrist_score
    
    def getNeckRULA(neck_angle : float , neck_twisted : int , neck_side_bending : int) :
        angle_limits = [[0, 10], [10, 20], [20 , 100], [0 , -100]]
        neck_score = neck_twisted + neck_side_bending

        # scoring for neck score
        if neck_angle >= angle_limits[0][0] and neck_angle <= angle_limits[0][1]:
            neck_score += 1
        elif neck_angle > angle_limits[1][0] and neck_angle <= angle_limits[1][1]:
            neck_score += 2
        elif neck_angle > angle_limits[2][0] and neck_angle <= angle_limits[2][1]:
            neck_score += 3
        elif neck_angle > angle_limits[3][0] and neck_angle <= angle_limits[3][1]:
            neck_score += 4

        return neck_score
    
    def getTrunkREBA(trunk_angle : float , trunk_twisted : int , trunk_side_bending : int):
        angle_limits = [[0], [0, 20], [20, 60], [60 , 180]]
        trunk_score = trunk_twisted + trunk_side_bending

        # scoring for trunk score 
        if trunk_angle == angle_limits[0][0]:
            trunk_score += 1
        elif trunk_angle > angle_limits[1][0] and trunk_angle <= angle_limits[1][1]:
            trunk_score += 2
        elif trunk_angle > angle_limits[2][0] and trunk_angle <= angle_limits[2][1]:
            trunk_score += 3
        elif trunk_angle > angle_limits[3][0] and trunk_angle <= angle_limits[2][1]:
            trunk_score += 4

        return trunk_score
    
    def getLegREBA(leg_feet_supported : int) :
        leg_score = leg_feet_supported
        return leg_score
    
    def get_Table_A_Score(upperArm_score : float, lowerArm_score : float, wrist_score : float , suppination : float ) :
        scores = np.round([upperArm_score, lowerArm_score, wrist_score, suppination])
        
        scores = scores.astype(int)

        table = np.genfromtxt("./RULA_tables/tableA.csv", delimiter=',', dtype=int)
        print("WS:" ,wrist_score)
        # generating x-coordinate
        if scores[0] == 1  and scores[1] == 1:
            x = 0
        elif scores[0] == 1 and scores[1] == 2:
            x = 1
        elif scores[0] == 1 and scores[1] == 3:
            x = 2
        elif scores[0] == 2 and scores[1] == 1:
            x = 3
        elif scores[0] == 2 and scores[1] == 2:
            x = 4
        elif scores[0] == 2 and scores[1] == 3:
            x = 5
        elif scores[0] == 3 and scores[1] == 1:
            x = 6
        elif scores[0] == 3 and scores[1] == 2:
            x = 7
        elif scores[0] == 3 and scores[1] == 3:
            x = 8
        elif scores[0] == 4 and scores[1] == 1:
            x = 9
        elif scores[0] == 4 and scores[1] == 2:
            x = 10
        elif scores[0] == 4 and scores[1] == 3:
            x = 11
        elif scores[0] == 5 and scores[1] == 1:
            x = 12
        elif scores[0] == 5 and scores[1] == 2:
            x = 13
        elif scores[0] == 5 and scores[1] == 3:
            x = 14
        elif scores[0] == 6 and scores[1] == 1:
            x = 15
        elif scores[0] == 6 and scores[1] == 2:
            x = 16
        elif scores[0] == 6 and scores[1] == 3:
            x = 17
        else:
            x = 0
            print("Error!x-value is invalid!")
                

        # generating y-coordinate
        if scores[2] == 1 and scores[3] == 1:
            y = 0
        elif scores[2] == 1 and scores[3] == 2:
            y = 1
        elif scores[2] == 2 and scores[3] == 1:
            y = 2
        elif scores[2] == 2 and scores[3] == 2:
            y = 3
        elif scores[2] == 3 and scores[3] == 1:
            y = 4
        elif scores[2] == 3 and scores[3] == 2:
            y = 5
        elif scores[2] == 4 and scores[3] == 1:
            y = 6
        elif scores[2] == 4 and scores[3] == 2:
            y = 7

        print("Score Tabelle A:")
        print("x: " + str(x) + "; y: " + str(y))
        print(str(table[x, y]))
        return table[x, y]
    
    def get_Table_B_Score(neck_score : float, trunk_score : float, leg_score : float) :
        scores = np.round([neck_score, trunk_score, leg_score])
        scores = scores.astype(int)

        table = np.genfromtxt("./RULA_tables/tableB.csv", delimiter=',', dtype=int)
        print('Neck_score' , scores[0])
        print('Trunk_score' , scores[1])
        print('Leg_score' , scores[2])

        # generating x-coordinates
        if scores[0] == 1:
            x = 0
        elif scores[0] == 2:
            x = 1
        elif scores[0] == 3:
            x = 2
        elif scores[0] == 4:
            x = 3
        elif scores[0] == 5:
            x = 4
        elif scores[0] == 6:
            x = 5

        # generating y-coordinates
        if scores[1] == 1 and scores[2] == 1:
            y = 0  
        elif scores[1] == 1 and scores[2] == 2:
            y = 1
        elif scores[1] == 2 and scores[2] == 1:
            y = 2
        elif scores[1] == 2 and scores[2] == 2:
            y = 3
        elif scores[1] == 3 and scores[2] == 1:
            y = 4
        elif scores[1] == 3 and scores[2] == 2:
            y = 5
        elif scores[1] == 4 and scores[2] == 1:
            y = 6
        elif scores[1] == 4 and scores[2] == 2:
            y = 7
        elif scores[1] == 5 and scores[2] == 1:
            y = 8
        elif scores[1] == 5 and scores[2] == 2:
            y = 9
        elif scores[1] == 6 and scores[2] == 1:
            y = 10
        elif scores[1] == 6 and scores[2] == 2:
            y = 11


        print("Score Tabelle B:")
        print("x: " + str(x) + "; y: " + str(y))
        print(str(table[x, y]))
        return table[x, y]
    

    def get_Table_C_Score(table_a_score , table_b_score) :
        """
        generates the final Rula Score based, on values of upper_table and lower_table
        """
        scores = np.round([table_a_score, table_b_score])
        scores = scores.astype(int)

        table = np.genfromtxt("./RULA_tables/tableC.csv", delimiter=',', dtype=int)

        # generating x-coordinate
        if scores[0] == 1:
            x = 0
        elif scores[0] == 2:
            x = 1
        elif scores[0] == 3:
            x = 2
        elif scores[0] == 4:
            x = 3
        elif scores[0] == 5:
            x = 4
        elif scores[0] == 6:
            x = 5
        elif scores[0] == 7:
            x = 6
        elif scores[0] >= 8:
            x = 7

        # generating y-coordinate
        if scores[1] == 1:
            y = 0
        elif scores[1] == 2:
            y = 1
        elif scores[1] == 3:
            y = 2
        elif scores[1] == 4:
            y = 3
        elif scores[1] == 5:
            y = 4
        elif scores[1] == 6:
            y = 5
        elif scores[1] >= 7:
            y = 6


        print("Score Tabelle C:")
        print("x: " + str(x) + "; y: " + str(y))
        print("==============")
        print("FINAL SCORE:")
        print("\t" + str(table[x, y]))
        print("==============")
        return table[x, y]

