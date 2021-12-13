use rand::prelude::*;
use rand::prelude::*;
use std::f64::consts::PI;
use std::ops::Add;
use std::ptr::addr_of_mut;

use super::world::*;

#[derive(Debug)]
pub struct Creature {
    pub age: i16,
    last_move: Option<Direction>,
    pub position: Position,
    pub neurons: Vec<Neuron>,
    pub connections: Vec<(usize, usize, f64)>,
    pub alive: bool,
    chromosomes: Vec<u32>,
    internal_neurons_count: u8,
    pub responsiveness: f64,
}

impl Creature {
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

    fn flip_random_bit_old(chromosome: u32) -> u32 {
        let bit_index = thread_rng().gen::<u32>() as usize % 32;
        let mask = 1 << bit_index;

        chromosome ^ mask
    }

    pub fn reproduce(&self, partner: &Creature) -> Creature {
        let mut chromosomes = vec![];
        for (c1, c2) in self.chromosomes.iter().zip(partner.chromosomes.iter()) {
            let c = if thread_rng().gen::<u8>() % 2 == 0 {
                *c1
            } else {
                *c2
            };

            let mc = if thread_rng().gen::<u8>() <= 2 {
                Creature::flip_random_bit(c)
            } else {
                c
            };

            chromosomes.push(mc);
        }

        Creature::from_chromosomes(chromosomes, self.internal_neurons_count)
    }

    pub fn from_chromosomes(chromosomes: Vec<u32>, internal_neurons_count: u8) -> Creature {
        let mut conns = chromosomes
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

            /*if last_input_ix == input_ix && last_output_ix == output_ix {
                if let Some((_, _, w)) = connections.pop() {
                    connections.push((input_ix, output_ix, weight as f64 / 16000. + w))
                }
            } else {*/
            unsafe {
                connections.push((input_ix, output_ix, weight as f64 / 8000. - 4.));
                //connections.push((addr_of_mut!(neurons[input_ix]), addr_of_mut!(neurons[output_ix]), weight as f64 / 8000. - 4.));
            }
            //}

            last_input_ix = input_ix;
            last_output_ix = output_ix;
        }

        Creature {
            age: 0,
            position: Position { x: 0, y: 0 },
            neurons: neurons,
            connections: connections,
            last_move: None,
            alive: true,
            chromosomes,
            internal_neurons_count,
            responsiveness: 0.5,
        }
    }

    pub fn set_position(&mut self, position: Position) {
        self.last_move = Direction::from_movement_array([
            position.x - self.position.x,
            position.y - self.position.y,
        ]);
        self.position = position;
    }

    pub fn act(&self, world: &World) -> Vec<ActionResult> {
        let mut add = vec![0.; self.neurons.len()];

        for connection in self.connections.iter() {
            //let neuron = unsafe { *connection.0 };
            let neuron = &self.neurons[connection.0];

            add[connection.1] += match &neuron {
                Neuron::Sensor(s) => self.get_sensor(s, world) * connection.2,
                Neuron::Internal(_) => neuron.activate(add[connection.0]) * connection.2,
                Neuron::Action(_) => unreachable!(),
            }
        }

                let xxxxx = add
                    .into_iter()
                    .zip(&self.neurons)
                    .filter_map(|(add, n)| {
                        if let Neuron::Action(a) = n {
                            Some((n.activate(add), a))
                        } else {
                            None
                        }
                    });

                let mut move_x = 0.;
                let mut move_y = 0.;
                let mut kill = 0.;
                let mut responsiveness = None;
                for (intensity, action) in xxxxx {
                    match action {
                        ActionType::Move(d) => {
                            move_x += d.x() * intensity;
                            move_y += d.y() * intensity;
                        }
                        ActionType::MoveForward if self.last_move.is_some() => {
                            move_x += self.last_move.as_ref().unwrap().x() * intensity;
                            move_y += self.last_move.as_ref().unwrap().y() * intensity;
                        }
                        ActionType::Kill if self.last_move.is_some() => {
                            kill += intensity
                        }
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
                            responsiveness = Some((intensity.tanh() + 1.) / 2.);
                        }
                        ActionType::MoveForward
                        | ActionType::Kill
                        | ActionType::MoveLeft
                        | ActionType::MoveRight => { },
                    }
                }
                /*
                let (kill, x, y, responsiveness) = xxxxx
                    .into_iter()
                    .fold(
                        (0., 0., 0., None),
                        |(kill, move_x, move_y, responsiveness), (intensity, action)| match action {
                            ActionType::Move(d) => (
                                kill,
                                move_x + d.x() * intensity,
                                move_y + d.y() * intensity,
                                responsiveness,
                            ),
                            ActionType::MoveForward if self.last_move.is_some() => (
                                kill,
                                move_x + self.last_move.as_ref().unwrap().x() * intensity,
                                move_y + self.last_move.as_ref().unwrap().y() * intensity,
                                responsiveness,
                            ),
                            ActionType::Kill if self.last_move.is_some() => {
                                (kill + intensity, move_x, move_y, responsiveness)
                            }
                            ActionType::MoveLeft if self.last_move.is_some() => {
                                let rotated = self.last_move.as_ref().unwrap().left();

                                (
                                    kill,
                                    move_x + rotated.x() * intensity,
                                    move_y + rotated.y() * intensity,
                                    responsiveness,
                                )
                            }
                            ActionType::MoveRight if self.last_move.is_some() => {
                                let rotated = self.last_move.as_ref().unwrap().right();

                                (
                                    kill,
                                    move_x + rotated.x() * intensity,
                                    move_y + rotated.y() * intensity,
                                    responsiveness,
                                )
                            }
                            ActionType::MoveRandom => {
                                let mv = Direction::random();

                                (
                                    kill,
                                    mv.x() * intensity,
                                    mv.y() * intensity,
                                    responsiveness,
                                )
                            }
                            ActionType::SetResponsiveness => {
                                (kill, move_x, move_y, Some((intensity.tanh() + 1.) / 2.))
                            }
                            ActionType::MoveForward
                            | ActionType::Kill
                            | ActionType::MoveLeft
                            | ActionType::MoveRight => (kill, move_x, move_y, responsiveness),
                        },
                    );*/

        let (normalized_kill, px, py) = (
            kill.tanh(),
            move_x.tanh() * self.responsiveness,
            move_y.tanh() * self.responsiveness,
        );

        let mut results = vec![];
        if let Some(r) = responsiveness {
            results.push(ActionResult::SetResponsiveness(self.position.clone(), r));
        }

        if normalized_kill > 10000. {
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

    fn get_sensor(&self, sensor_type: &SensorType, world: &World) -> f64 {
        match sensor_type {
            SensorType::Age => (world.age - 125) as f64 / 125.,
            SensorType::NorthBoundaryDistance => 1. - self.position.y as f64 / world.height as f64,
            SensorType::EastBoundaryDistance => 1. - self.position.x as f64 / world.width as f64,
            SensorType::SouthBoundaryDistance => self.position.y as f64 / world.height as f64,
            SensorType::WestBoundaryDistance => self.position.x as f64 / world.width as f64,
            //SensorType::PositionX => self.position.x as f64,
            //SensorType::PositionY => self.position.y as f64,
            SensorType::Density => world.get_neighborhood(&self.position, 3) as f64 / 8.,
            SensorType::Random => thread_rng().gen::<f64>(),
            SensorType::Barrier => 0.,
            SensorType::Oscillator => (world.age as f64 / 10.).sin(),
            SensorType::DistanceCreatureForward => match self.last_move {
                Some(m) => {
                    (world.distance_next_creature(&self.position, &m, 20) as f64 - 20.).tanh()
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

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Neuron {
    Sensor(SensorType),
    Internal(u8),
    Action(ActionType),
}

impl Neuron {
    #[inline]
    fn activate(&self, input: f64) -> f64 {
        input.tanh()
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SensorType {
    Age,
    NorthBoundaryDistance,
    EastBoundaryDistance,
    SouthBoundaryDistance,
    WestBoundaryDistance,
    //PositionX,
    //PositionY,
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
            //10 => SensorType::PositionX,
            //11 => SensorType::PositionY,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ActionType {
    Move(Direction),
    MoveForward,
    MoveLeft,
    MoveRight,
    MoveRandom,
    SetResponsiveness,
    Kill,
}

impl From<u8> for ActionType {
    fn from(item: u8) -> ActionType {
        match item % 14 {
            0 => ActionType::Move(Direction::North),
            1 => ActionType::Move(Direction::NorthEast),
            2 => ActionType::Move(Direction::East),
            3 => ActionType::Move(Direction::SouthEast),
            4 => ActionType::Move(Direction::South),
            5 => ActionType::Move(Direction::SouthWest),
            6 => ActionType::Move(Direction::West),
            7 => ActionType::Move(Direction::NorthWest),
            8 => ActionType::MoveForward,
            9 => ActionType::MoveLeft,
            10 => ActionType::MoveRight,
            11 => ActionType::MoveRandom,
            12 => ActionType::SetResponsiveness,
            13 => ActionType::Kill,
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
