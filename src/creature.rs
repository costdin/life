use super::world::*;
use rand::prelude::*;
use std::ops::Add;
use std::sync::Arc;
use std::sync::Mutex;

const CHROMOSOME_LIMIT: usize = 200;
const NEURON_LIMIT: usize = CHROMOSOME_LIMIT * 2;

#[derive(Debug, Clone)]
pub struct Creature<const CHROMOSOME_COUNT: usize> {
    last_move: Option<Direction>,
    pub position: Position,
    pub neurons: Vec<Neuron>,
    pub connections: Vec<(usize, usize, f64)>,
    pub alive: bool,
    chromosomes: [u32; CHROMOSOME_COUNT],
    internal_neurons_count: u8,
    pub responsiveness: Arc<Mutex<f64>>,
    pub oscillator_frequency: Arc<Mutex<f64>>,
}

impl<const CHROMOSOME_COUNT: usize> Creature<CHROMOSOME_COUNT> {
    fn flip_random_bit(chromosome: u32) -> u32 {
        let p = thread_rng().gen::<f64>();

        if p < 0.2 {
            chromosome ^ 0x80000000u32
        } else if p < 0.4 {
            chromosome ^ 0x00800000u32
        } else if p < 0.6 {
            chromosome ^ (1 << ((thread_rng().gen::<u32>() as usize % 7) + 24))
        } else if p < 0.8 {
            chromosome ^ (1 << ((thread_rng().gen::<u32>() as usize % 7) + 16))
        } else {
            chromosome ^ (1 << thread_rng().gen::<u32>() as usize % 16)
        }
    }

    pub fn reproduce(&self, partner: &Creature<CHROMOSOME_COUNT>) -> Creature<CHROMOSOME_COUNT> {
        let mut chromosomes = [0; CHROMOSOME_COUNT];
        for (i, (c1, c2)) in self.chromosomes.iter().zip(partner.chromosomes.iter()).enumerate() {
            let c = if thread_rng().gen::<u8>() % 2 == 0 {
                *c1
            } else {
                *c2
            };

            //let mc = if thread_rng().gen::<u8>() <= 2 {
            let mc = if thread_rng().gen::<f32>() <= 0.001 {
                Creature::<CHROMOSOME_COUNT>::flip_random_bit(c)
            } else {
                c
            };

            chromosomes[i] = mc;
        }

        Creature::from_chromosomes(chromosomes, self.internal_neurons_count)
    }

    #[inline]
    fn input_neuron(chromosome: &u32) -> u32 {
        chromosome >> 24 & 0xFF
    }

    #[inline]
    fn output_neuron(chromosome: &u32) -> u32 {
        chromosome >> 16 & 0xFF
    }

    pub fn from_chromosomes(mut chromosomes: [u32; CHROMOSOME_COUNT], internal_neurons_count: u8) -> Creature<CHROMOSOME_COUNT> {
        if CHROMOSOME_COUNT > CHROMOSOME_LIMIT {
            panic!("Limit the number of chromosomes to {CHROMOSOME_LIMIT} for performance reasons (once generic_const_exprs is stabilized, we'll re-evaluate)")
        }

        chromosomes.sort();

        let pruned_chromosomes = chromosomes
            .iter()
            .filter(|c| {
                *c & 0x00800000 > 0
                    || chromosomes.iter().any(|d| {
                        Creature::<CHROMOSOME_COUNT>::output_neuron(c) == Creature::<CHROMOSOME_COUNT>::input_neuron(d)
                            && Creature::<CHROMOSOME_COUNT>::input_neuron(d) != Creature::<CHROMOSOME_COUNT>::output_neuron(d)
                    })
            })
            .map(|c| *c)
            .collect::<Vec<u32>>();

        let mut new_chromosomes = vec![];
        let mut last_c = 0x00007D00; // 0x00007D00 is 0 weight
        for c in pruned_chromosomes.iter() {
            if (c & 0xFFFF0000) == (last_c & 0xFFFF0000) {
                let v1 = last_c & 0xFFFF;
                let v2 = c & 0xFFFF;

                if v1 + v2 > 0xFFFF {
                    last_c = last_c | 0x0000FFFF;
                } else {
                    last_c = last_c | (v1 + v2);
                }
            } else {
                if last_c & 0x0000FFFF != 0x00007D00 {
                    new_chromosomes.push(last_c);
                }
                last_c = *c;
            }
        }
        new_chromosomes.push(last_c);

        let mut conns = new_chromosomes
            .iter()
            .map(|c| {
                let is_sensor = c & 0x80000000u32 > 0;
                let input_type = (0x7F & (c >> 24)) as u8;

                let input = if is_sensor {
                    Neuron::Sensor(input_type.into())
                } else {
                    Neuron::Internal(input_type % internal_neurons_count)
                };

                let is_action = c & 0x00800000u32 > 0;
                let output_type = (0x7F & (c >> 16)) as u8;

                let output = if is_action {
                    Neuron::Action(output_type.into())
                } else {
                    Neuron::Internal(output_type % internal_neurons_count)
                };

                let weight = c & 0xFFFF;

                (input, output, weight)
            })
            .collect::<Vec<_>>();

        conns.sort();

        let mut neurons = vec![];
        let mut connections = vec![];
        let mut last_input_ix = usize::MAX;
        let mut last_output_ix = usize::MAX;
        for (input, output, weight) in conns {
            let input_ix = neurons.iter().position(|n| n == &input).unwrap_or_else(|| {
                neurons.push(input);
                neurons.len() - 1
            });

            let output_ix = neurons
                .iter()
                .position(|n| n == &output)
                .unwrap_or_else(|| {
                    neurons.push(output);
                    neurons.len() - 1
                });

            //if last_input_ix == input_ix && last_output_ix == output_ix {
            //    if let Some((_, _, w)) = connections.pop() {
            //        connections.push((input_ix, output_ix, weight as f64 / 16000. + w))
            //    }
            //} else {
            connections.push((input_ix, output_ix, weight as f64 / 8000. - 4.));
            //}

            last_input_ix = input_ix;
            last_output_ix = output_ix;
        }

        Creature {
            position: Position { x: 0, y: 0 },
            neurons: neurons,
            connections: connections,
            last_move: None,
            alive: true,
            chromosomes,
            internal_neurons_count,
            responsiveness: Arc::new(Mutex::new(0.5)),
            oscillator_frequency: Arc::new(Mutex::new(1.)),
        }
    }

    pub fn remove_dead_neurons(&mut self) {
        let int = self
            .neurons
            .iter()
            .filter(|n| matches!(n, Neuron::Internal(_)))
            .collect::<Vec<_>>();
        // remove
        for c in self.connections.iter() {}
    }

    pub fn set_position(&mut self, position: Position) {
        self.last_move = Direction::from_movement_array([
            position.x - self.position.x,
            position.y - self.position.y,
        ]);
        self.position = position;
    }

    pub fn act(&self, world: &World<CHROMOSOME_COUNT>) -> Vec<ActionResult> {
        let mut add = [0.; NEURON_LIMIT];

        for connection in self.connections.iter() {
            //let neuron = unsafe { *connection.0 };
            let neuron = &self.neurons[connection.0];

            add[connection.1] += match &neuron {
                Neuron::Sensor(s) => self.get_sensor(s, world) * connection.2,
                n@ Neuron::Internal(_) => n.activate(add[connection.0]) * connection.2,
                Neuron::Action(_) => unreachable!(),
            }
        }

        let mut move_x = 0.;
        let mut move_y = 0.;
        let mut kill = 0.;
        let mut responsiveness = None;
        let mut new_freq = None;


        for i in 0..self.neurons.len() {
            if let Neuron::Action(action) = &self.neurons[i] {
                let intensity = self.neurons[i].activate(add[i]);
                match action {
                    ActionType::Move(d) => {
                        move_x += d.x() * intensity;
                        move_y += d.y() * intensity;
                    }
                    ActionType::MoveForward if self.last_move.is_some() => {
                        move_x += self.last_move.as_ref().unwrap().x() * intensity;
                        move_y += self.last_move.as_ref().unwrap().y() * intensity;
                    }
                    ActionType::Kill if self.last_move.is_some() => kill += intensity,
                    ActionType::MoveLeft if self.last_move.is_some() => {
                        let rotated = self.last_move.as_ref().unwrap().left();
    
                        move_x += rotated.x() * intensity;
                        move_y += rotated.y() * intensity;
                    }
                    ActionType::MoveRight if self.last_move.is_some() => {
                        let rotated = self.last_move.as_ref().unwrap().right();
    
                        move_x += rotated.x() * intensity;
                        move_y += rotated.y() * intensity;
                    }
                    ActionType::MoveRandom => {
                        let mv = Direction::random();
    
                        move_x += mv.x() * intensity;
                        move_x += mv.y() * intensity;
                    }
                    ActionType::SetResponsiveness => {
                        responsiveness = Some((logistic_thing(intensity) + 1.) / 2.);
                    }
                    ActionType::SetOscillator => {
                        new_freq = Some(logistic_thing(intensity) + 1.);
                    }
                    ActionType::MoveForward
                    | ActionType::Kill
                    | ActionType::MoveLeft
                    | ActionType::MoveRight => {}
                }
            }
        }

        let mut resp = self.responsiveness.lock().unwrap();
        let (normalized_kill, px, py) = (logistic_thing(kill), logistic_thing(move_x) * *resp, logistic_thing(move_y) * *resp);

        if let Some(r) = responsiveness {
            *resp = r;
        }

        if let Some(s) = new_freq {
            *self.oscillator_frequency.lock().unwrap() = s;
        }

        let mut results = vec![];
        if normalized_kill > 0.5 && normalized_kill > thread_rng().gen::<f64>() {
            let kill_position = &self.position + self.last_move.as_ref();

            if world.is_position_in_bounds(&kill_position) && !world.is_empty(&kill_position) {
                results.push(ActionResult::Kill(kill_position));
            }
        }

        let mx = if px.abs() > thread_rng().gen::<f64>() {
            px.signum() as i16
        } else {
            0
        };
        let my = if py.abs() > thread_rng().gen::<f64>() {
            py.signum() as i16
        } else {
            0
        };

        if (mx, my) != (0, 0) {
            let new_position = &self.position + [mx, my];

            if world.is_position_in_bounds(&new_position) && world.is_empty(&new_position) {
                results.push(ActionResult::Move(self.position.clone(), new_position));
            }
        }

        results
    }

    fn get_sensor(&self, sensor_type: &SensorType, world: &World<CHROMOSOME_COUNT>) -> f64 {
        match sensor_type {
            SensorType::Age => (world.age_f - 125.) / 125.,
            SensorType::NorthBoundaryDistance => 1. - self.position.y as f64 / world.height_f,
            SensorType::EastBoundaryDistance => 1. - self.position.x as f64 / world.width_f,
            SensorType::SouthBoundaryDistance => self.position.y as f64 / world.height_f,
            SensorType::WestBoundaryDistance => self.position.x as f64 / world.width_f,
            SensorType::Density => world.get_neighborhood(&self.position, 3) as f64 / 8.,
            SensorType::Random => thread_rng().gen::<f64>(),
            SensorType::Barrier => 0.,
            SensorType::Oscillator => {
                (*self.oscillator_frequency.lock().unwrap() * world.age_f / 10.).sin()
            }
            SensorType::DistanceCreatureForward => match self.last_move {
                Some(m) => {
                    logistic_thing(world.distance_next_creature(&self.position, &m, 20) as f64 - 20.)
                }
                None => 0.,
            },
        }
    }
}

#[derive(Debug)]
pub enum ActionResult {
    Kill(Position),
    Move(Position, Position),
    SetResponsiveness(Position, f64),
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum Neuron {
    Sensor(SensorType),
    Internal(u8),
    Action(ActionType),
}

impl Neuron {
    #[inline]
    fn activate(&self, input: f64) -> f64 {
        logistic_thing(input)
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum SensorType {
    Age,
    NorthBoundaryDistance,
    EastBoundaryDistance,
    SouthBoundaryDistance,
    WestBoundaryDistance,
    Density,
    Random,
    Barrier,
    Oscillator,
    DistanceCreatureForward,
}

impl From<u8> for SensorType {
    fn from(item: u8) -> SensorType {
        match item % 10 {
            0 => SensorType::Age,
            1 => SensorType::NorthBoundaryDistance,
            2 => SensorType::EastBoundaryDistance,
            3 => SensorType::SouthBoundaryDistance,
            4 => SensorType::WestBoundaryDistance,
            5 => SensorType::Density,
            6 => SensorType::Random,
            7 => SensorType::Barrier,
            8 => SensorType::Oscillator,
            9 => SensorType::DistanceCreatureForward,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum ActionType {
    Move(Direction),
    MoveForward,
    MoveLeft,
    MoveRight,
    MoveRandom,
    SetResponsiveness,
    SetOscillator,
    Kill,
}

impl From<u8> for ActionType {
    fn from(item: u8) -> ActionType {
        match item % 11 {
            0 => ActionType::Move(Direction::North),
            1 => ActionType::Move(Direction::East),
            2 => ActionType::Move(Direction::South),
            3 => ActionType::Move(Direction::West),
            4 => ActionType::MoveForward,
            5 => ActionType::MoveLeft,
            6 => ActionType::MoveRight,
            7 => ActionType::MoveRandom,
            8 => ActionType::SetResponsiveness,
            9 => ActionType::SetOscillator,
            10 => ActionType::Kill,
            /*
            1 => ActionType::Move(Direction::NorthEast),
            3 => ActionType::Move(Direction::SouthEast),
            5 => ActionType::Move(Direction::SouthWest),
            7 => ActionType::Move(Direction::NorthWest),
            */
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Position {
    pub x: i16,
    pub y: i16,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub enum Direction {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

impl Direction {
    pub fn left(&self) -> Direction {
        match self {
            Direction::North => Direction::East,
            Direction::NorthEast => Direction::SouthEast,
            Direction::East => Direction::South,
            Direction::SouthEast => Direction::SouthWest,
            Direction::South => Direction::West,
            Direction::SouthWest => Direction::NorthWest,
            Direction::West => Direction::North,
            Direction::NorthWest => Direction::NorthEast,
        }
    }

    pub fn right(&self) -> Direction {
        match self {
            Direction::East => Direction::North,
            Direction::SouthEast => Direction::NorthEast,
            Direction::South => Direction::East,
            Direction::SouthWest => Direction::SouthEast,
            Direction::West => Direction::South,
            Direction::NorthWest => Direction::SouthWest,
            Direction::North => Direction::West,
            Direction::NorthEast => Direction::NorthWest,
        }
    }

    pub fn x(&self) -> f64 {
        match self {
            Direction::East | Direction::NorthEast | Direction::NorthWest => 1.,
            Direction::West | Direction::SouthEast | Direction::SouthWest => -1.,
            _ => 0.,
        }
    }

    pub fn y(&self) -> f64 {
        match self {
            Direction::North | Direction::NorthEast | Direction::SouthEast => 1.,
            Direction::South | Direction::NorthWest | Direction::SouthWest => -1.,
            _ => 0.,
        }
    }

    pub fn movement_array(&self) -> [i16; 2] {
        match self {
            Direction::North => [0, 1],
            Direction::NorthEast => [1, 1],
            Direction::East => [1, 0],
            Direction::SouthEast => [1, -1],
            Direction::South => [0, -1],
            Direction::SouthWest => [-1, -1],
            Direction::West => [-1, 0],
            Direction::NorthWest => [-1, 1],
        }
    }

    pub fn from_movement_array(item: [i16; 2]) -> Option<Direction> {
        match item {
            [0, 1] => Some(Direction::North),
            [1, 1] => Some(Direction::NorthEast),
            [1, 0] => Some(Direction::East),
            [1, -1] => Some(Direction::SouthEast),
            [0, -1] => Some(Direction::South),
            [-1, -1] => Some(Direction::SouthWest),
            [-1, 0] => Some(Direction::West),
            [-1, 1] => Some(Direction::NorthWest),
            _ => None,
        }
    }

    pub fn random() -> Direction {
        match thread_rng().gen::<u8>() % 8 {
            0 => Direction::North,
            1 => Direction::NorthEast,
            2 => Direction::East,
            3 => Direction::SouthEast,
            4 => Direction::South,
            5 => Direction::SouthWest,
            6 => Direction::West,
            7 => Direction::NorthWest,
            _ => unreachable!(),
        }
    }
}

impl Add<Option<&Direction>> for &Position {
    type Output = Position;

    fn add(self, other: Option<&Direction>) -> Position {
        match other {
            None => Position {
                x: self.x,
                y: self.y,
            },
            Some(o) => self.add_direction(o),
        }
    }
}

impl Position {
    pub fn add_direction(&self, direction: &Direction) -> Position {
        let [x, y] = direction.movement_array();

        Position {
            x: self.x + x,
            y: self.y + y,
        }
    }
}

/*
impl Add<Direction> for &Position {
    type Output = Position;

    fn add(self, other: Direction) -> Position {
        match other {
            Direction::North => Position {
                x: self.x,
                y: self.y + 1,
            },
            Direction::NorthEast => Position {
                x: self.x + 1,
                y: self.y + 1,
            },
            Direction::East => Position {
                x: self.x + 1,
                y: self.y,
            },
            Direction::SouthEast => Position {
                x: self.x + 1,
                y: self.y - 1,
            },
            Direction::South => Position {
                x: self.x,
                y: self.y - 1,
            },
            Direction::SouthWest => Position {
                x: self.x - 1,
                y: self.y - 1,
            },
            Direction::West => Position {
                x: self.x - 1,
                y: self.y,
            },
            Direction::NorthWest => Position {
                x: self.x - 1,
                y: self.y + 1,
            },
        }
    }
}
*/
impl Add<[i16; 2]> for &Position {
    type Output = Position;

    fn add(self, other: [i16; 2]) -> Position {
        Position {
            x: self.x + other[0],
            y: self.y + other[1],
        }
    }
}

#[inline]
fn logistic_thing(n: f64) -> f64 {
    n / (1. + n.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chromosomes_base() {
        let c1 = 0x8080FF00u32;
        let c2 = 0x808100FFu32;

        let creature = Creature::from_chromosomes([c1, c2], 1);

        assert_eq!(creature.neurons.len(), 3);
        match (&creature.neurons[0], &creature.neurons[1]) {
            (
                Neuron::Sensor(SensorType::Age),
                Neuron::Action(ActionType::Move(Direction::North)),
            ) => {
                assert_eq!(creature.connections[0], (0, 1, 0xFF00 as f64 / 8000. - 4.));
                assert_eq!(creature.connections[1], (0, 2, 0x00FF as f64 / 8000. - 4.));
            }
            _ => assert_eq!(1, 0),
        }
    }

    #[test]
    fn chromosomes_are_compressed() {
        let c1 = 0x8080FF00u32;
        let c2 = 0x808000FFu32;

        let creature = Creature::from_chromosomes([c1, c2], 1);

        assert_eq!(creature.neurons.len(), 2);
        match (&creature.neurons[0], &creature.neurons[1]) {
            (
                Neuron::Sensor(SensorType::Age),
                Neuron::Action(ActionType::Move(Direction::North)),
            ) => {
                assert_eq!(creature.connections[0], (0, 1, 0xFFFF as f64 / 8000. - 4.))
            }
            _ => assert_eq!(1, 0),
        }
    }

    #[test]
    fn chromosomes_with_internal_neurons() {
        let c1 = 0x8000FF00u32;
        let c2 = 0x008000FFu32;

        let creature = Creature::from_chromosomes([c1, c2], 1);

        assert_eq!(creature.neurons.len(), 3);
        match (&creature.neurons[0], &creature.neurons[1]) {
            (Neuron::Sensor(SensorType::Age), Neuron::Internal(0)) => {
                assert_eq!(creature.connections[0], (0, 1, 0xFF00 as f64 / 8000. - 4.))
            }
            _ => assert_eq!(1, 0),
        }

        match (&creature.neurons[1], &creature.neurons[2]) {
            (Neuron::Internal(0), Neuron::Action(ActionType::Move(Direction::North))) => {
                assert_eq!(creature.connections[1], (1, 2, 0x00FF as f64 / 8000. - 4.))
            }
            _ => assert_eq!(1, 0),
        }
    }

    #[test]
    fn unused_neurons_are_pruned() {
        let c1 = 0x8000FF00u32;
        let c2 = 0x808000FFu32;

        let creature = Creature::from_chromosomes([c1, c2], 1);

        assert_eq!(creature.neurons.len(), 2);
        match (&creature.neurons[0], &creature.neurons[1]) {
            (
                Neuron::Sensor(SensorType::Age),
                Neuron::Action(ActionType::Move(Direction::North)),
            ) => {
                assert_eq!(creature.connections[0], (0, 1, 0x00FF as f64 / 8000. - 4.))
            }
            _ => assert_eq!(1, 0),
        }
    }

    #[test]
    fn unused_neurons_with_loop_are_pruned() {
        let c1 = 0x8000FF00u32;
        let c2 = 0x808000FFu32;
        let c3 = 0x0000FF00u32;

        let creature = Creature::from_chromosomes([c1, c2, c3], 1);

        assert_eq!(creature.neurons.len(), 2);
        match (&creature.neurons[0], &creature.neurons[1]) {
            (
                Neuron::Sensor(SensorType::Age),
                Neuron::Action(ActionType::Move(Direction::North)),
            ) => {
                assert_eq!(creature.connections[0], (0, 1, 0x00FF as f64 / 8000. - 4.))
            }
            _ => assert_eq!(1, 0),
        }
    }

    #[test]
    fn xxx() {
        let chromosomes = [
            0x04F29154, //83005780,
            0x174C423B, //390873659,
            0x25CE2378, //634266488,
            0x2EAF5007, //783241223,
            0x43714DFC, //1131499004,
            0x591897A1, //1494783905,
            0x6E2A0660, //1848247904,
            0x7C796C04, //2088332292,
            0x8DBACD0B, //2377829643,
            0xA0D0E7D2, //2698045394,
            0xA9811468, //2843808872,
            0xBFAFF28D, //3215979149,
            0xC2E0FD24, //3269524772,
            0xDBAFFA18, //3685743128,
            0xF4230EF8, //4095938296,
            0xF4AD27D7, //4104988631
        ];

        let c = Creature::from_chromosomes(chromosomes, 4);

        println!("{:#?}", c);
        println!("==================================================");
        println!("{:#?}", c.neurons);
        println!("==================================================");
        println!("{:#?}", c.connections);

        assert_eq!(0, 1);
    }
}
