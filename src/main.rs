use rand::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::ops::Add;

const BOX_HEIGHT: i16 = 1000;
const BOX_WIDTH: i16 = 1000;
const GENOME_SIZE: usize = 16;
const INTERNAL_NEURONS_COUNT: u8 = 4;

fn main() {
    let creatures = (0..2000 * GENOME_SIZE)
        .map(|_| thread_rng().gen::<u32>())
        .collect::<Vec<_>>()
        .chunks(GENOME_SIZE)
        .map(|c| Creature::from_chromosomes(c.to_vec(), INTERNAL_NEURONS_COUNT))
        .collect::<Vec<_>>();

    let mut world = World::new(creatures, BOX_HEIGHT as usize, BOX_WIDTH as usize);

    for generation in 0..10 {
        for _ in 0..100 {
            world.step();
        }
        let killed = world.creatures.iter().filter(|c| !c.alive).count();
        let filtered = world
            .creatures
            .iter()
            .filter(|c| {
                c.alive && (c.position.x >= BOX_WIDTH / 2 || c.position.y >= BOX_HEIGHT / 2)
            })
            .count();
        let survived = world
            .creatures
            .iter()
            .filter(|c| c.alive && c.position.x < BOX_WIDTH / 2 && c.position.y < BOX_HEIGHT / 2)
            .collect::<Vec<_>>();

        println!(
            "{}: {} survived ({} killed, {} filtered)",
            generation,
            survived.len(),
            killed,
            filtered
        );

        let new_creatures = (0..2000)
            .map(|_| {
                (
                    thread_rng().gen::<u32>() as usize % survived.len(),
                    thread_rng().gen::<u32>() as usize % survived.len(),
                )
            })
            .map(|(p1, p2)| survived[p1].reproduce(survived[p2]))
            .collect::<Vec<_>>();

        world = World::new(new_creatures, BOX_HEIGHT as usize, BOX_WIDTH as usize);
    }
}

struct World {
    creatures: Vec<Creature>,
    grid: Vec<Option<usize>>,
    width: usize,
    height: usize,
}

impl World {
    pub fn new(mut creatures: Vec<Creature>, height: usize, width: usize) -> World {
        let mut rng = rand::thread_rng();
        let size = width * height;

        let mut grid: Vec<Option<usize>> = (0..size).step_by(1).map(|_| None).collect();

        for (ix, creature) in creatures.iter_mut().enumerate() {
            let mut i = rng.next_u32() as usize % size;

            while !grid[i].is_none() {
                i = rng.next_u32() as usize % size;
            }

            creature.position = Position {
                x: (i % height) as i16,
                y: (i / height) as i16,
            };
            grid[i] = Some(ix);
        }

        World {
            creatures,
            grid,
            width,
            height,
        }
    }

    pub fn step(&mut self) {
        let (kills, moves) = self
            .creatures
            .par_iter()
            .filter(|c| c.alive)
            .map(|c| c.act(self))
            .flatten()
            .fold(
                || (vec![], vec![]),
                |(mut kills, mut moves), item| match item {
                    ActionResult::Move(src, dst) => {
                        moves.push((src, dst));
                        (kills, moves)
                    }
                    ActionResult::Kill(k) => {
                        kills.push(k);
                        (kills, moves)
                    }
                },
            )
            .reduce(
                || (vec![], vec![]),
                |(mut kills, mut moves), (k, m)| {
                    kills.extend_from_slice(&k);
                    moves.extend_from_slice(&m);

                    (kills, moves)
                },
            );

        for kill in kills {
            let i = self.position_to_grid_index(&kill);
            if let Some(ix) = self.grid[i] {
                self.creatures[ix].alive = false;
                self.grid[i] = None;
            }
        }

        for (source, destination) in moves {
            let dst = self.position_to_grid_index(&destination);
            if self.grid[dst].is_none() {
                let src = self.position_to_grid_index(&source);

                if let Some(ix) = self.grid[src] {
                    if self.creatures[ix].alive {
                        self.grid[src] = None;
                        self.grid[dst] = Some(ix);
                        self.creatures[ix].set_position(destination);
                    }
                }
            }
        }
    }

    #[inline]
    fn position_to_grid_index(&self, position: &Position) -> usize {
        self.get_grid_index(position.x as usize, position.y as usize)
    }

    #[inline]
    fn get_grid_index(&self, x: usize, y: usize) -> usize {
        x + y * self.width
    }

    #[inline]
    pub fn is_empty(&self, position: &Position) -> bool {
        self.grid[self.position_to_grid_index(position)].is_none()
    }

    #[inline]
    pub fn is_position_in_bounds(&self, position: &Position) -> bool {
        position.x >= 0
            && position.y >= 0
            && self.is_in_bounds(position.x as usize, position.y as usize)
    }

    #[inline]
    pub fn is_in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    pub fn get_neighborhood(&self, position: &Position, radius: usize) -> usize {
        let minx = if position.x as usize > radius {
            position.x as usize - radius
        } else {
            0
        };
        let maxx = usize::min(self.width - 1, position.x as usize + radius);

        let miny = if position.y as usize > radius {
            position.y as usize - radius
        } else {
            0
        };
        let maxy = usize::min(self.height - 1, position.y as usize + radius);

        (minx..maxx)
            .zip(miny..maxy)
            .filter(|(x, y)| self.grid[self.get_grid_index(*x, *y)].is_some())
            .count()
    }
}

#[derive(Debug)]
struct Creature {
    pub age: i16,
    last_move: Option<Direction>,
    position: Position,
    neurons: Vec<Neuron>,
    connections: Vec<(usize, usize, f64)>,
    pub alive: bool,
    chromosomes: Vec<u32>,
    internal_neurons_count: u8,
}

impl Creature {
    fn flip_random_bit(mut chromosomes: Vec<u32>) -> Vec<u32> {
        let mut rng = rand::thread_rng();

        let chromosome_to_flip = rng.next_u32() as usize % chromosomes.len();
        let bit_index = rng.next_u32() as usize % 32;
        let mask = 1 << bit_index;

        chromosomes[chromosome_to_flip] ^= mask;

        chromosomes
    }

    pub fn reproduce(&self, partner: &Creature) -> Creature {
        let mut rng = rand::thread_rng();

        let mut chromosomes = vec![];
        for (c1, c2) in self.chromosomes.iter().zip(partner.chromosomes.iter()) {
            if rng.next_u32() % 2 == 0 {
                chromosomes.push(*c1);
            } else {
                chromosomes.push(*c2);
            };
        }

        // mutation
        if rng.next_u32() % 1000 == 0 {
            chromosomes = Creature::flip_random_bit(chromosomes);
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

            if last_input_ix == input_ix && last_output_ix == output_ix {
                if let Some((_, _, w)) = connections.pop() {
                    connections.push((input_ix, output_ix, weight as f64 / 16000. + w))
                }
            } else {
                connections.push((input_ix, output_ix, weight as f64 / 16000.));
            }

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
            let neuron = &self.neurons[connection.0];

            add[connection.1] += match &neuron {
                Neuron::Sensor(s) => self.get_sensor(s, world) * connection.2,
                Neuron::Internal(_) => neuron.activate(add[connection.0]) * connection.2,
                Neuron::Action(_) => unreachable!(),
            }
        }

        let (kill, x, y) = add
            .into_iter()
            .zip(&self.neurons)
            .filter_map(|(add, n)| {
                if let Neuron::Action(a) = n {
                    Some((n.activate(add), a))
                } else {
                    None
                }
            })
            .fold(
                (0., 0., 0.),
                |(kill, move_x, move_y), (intensity, action)| match action {
                    ActionType::Move(d) => {
                        (kill, move_x + d.x() * intensity, move_y + d.y() * intensity)
                    }
                    ActionType::MoveForward if self.last_move.is_some() => (
                        kill,
                        move_x + self.last_move.as_ref().unwrap().x() * intensity,
                        move_y + self.last_move.as_ref().unwrap().y() * intensity,
                    ),
                    ActionType::Kill if self.last_move.is_some() => {
                        (kill + intensity, move_x, move_y)
                    }
                    _ => (kill, move_x, move_y),
                },
            );

        let mut results = vec![];
        if kill > 0.5 {
            let kill_position = &self.position + self.last_move.as_ref();

            if world.is_position_in_bounds(&kill_position) && !world.is_empty(&kill_position) {
                results.push(ActionResult::Kill(kill_position));
            }
        }

        let new_position = &self.position
            + [
                if x.abs() > 0.5 { x.signum() as i16 } else { 0 },
                if y.abs() > 0.5 { y.signum() as i16 } else { 0 },
            ];

        if world.is_position_in_bounds(&new_position) && world.is_empty(&new_position) {
            results.push(ActionResult::Move(self.position.clone(), new_position));
        }

        results
    }

    fn get_sensor(&self, sensor_type: &SensorType, world: &World) -> f64 {
        match sensor_type {
            SensorType::Age => self.age as f64,
            SensorType::NorthBoundaryDistance => (BOX_HEIGHT - self.position.y) as f64,
            SensorType::EastBoundaryDistance => (BOX_WIDTH - self.position.x) as f64,
            SensorType::SouthBoundaryDistance => self.position.x as f64,
            SensorType::WestBoundaryDistance => self.position.y as f64,
            SensorType::PositionX => self.position.x as f64,
            SensorType::PositionY => self.position.y as f64,
            SensorType::Density => world.get_neighborhood(&self.position, 3) as f64,
            SensorType::Random => thread_rng().gen::<f64>(),
            SensorType::Barrier => 0.,
        }
    }
}

#[derive(Debug)]
enum ActionResult {
    Kill(Position),
    Move(Position, Position),
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Neuron {
    Sensor(SensorType),
    Internal(u8),
    Action(ActionType),
}

impl Neuron {
    fn activate(&self, input: f64) -> f64 {
        2. * input.atan() / PI
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SensorType {
    Age,
    NorthBoundaryDistance,
    EastBoundaryDistance,
    SouthBoundaryDistance,
    WestBoundaryDistance,
    PositionX,
    PositionY,
    Density,
    Random,
    Barrier,
}

impl From<u8> for SensorType {
    fn from(item: u8) -> SensorType {
        match item % 10 {
            0 => SensorType::Age,
            1 => SensorType::NorthBoundaryDistance,
            2 => SensorType::EastBoundaryDistance,
            3 => SensorType::SouthBoundaryDistance,
            4 => SensorType::WestBoundaryDistance,
            5 => SensorType::PositionX,
            6 => SensorType::PositionY,
            7 => SensorType::Density,
            8 => SensorType::Random,
            9 => SensorType::Barrier,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ActionType {
    Move(Direction),
    MoveForward,
    Kill,
}

impl From<u8> for ActionType {
    fn from(item: u8) -> ActionType {
        match item % 10 {
            0 => ActionType::Move(Direction::North),
            1 => ActionType::Move(Direction::NorthEast),
            2 => ActionType::Move(Direction::East),
            3 => ActionType::Move(Direction::SouthEast),
            4 => ActionType::Move(Direction::South),
            5 => ActionType::Move(Direction::SouthWest),
            6 => ActionType::Move(Direction::West),
            7 => ActionType::Move(Direction::NorthWest),
            8 => ActionType::MoveForward,
            9 => ActionType::Kill,
            _ => unreachable!(),
        }
    }
}

/*
impl ActionType {
    pub fn to_action_result(&self, creature: &Creature) -> ActionResult {
        match self {
            ActionType::Move(d) => ActionResult::Move(creature.position + *d),
            ActionType::MoveForward => ActionResult::Move(creature.position + creature.last_move),
            ActionType::Kill => ActionResult::Kill(creature.position + creature.last_move),
        }
    }
}
*/

#[derive(Debug, Clone)]
struct Position {
    pub x: i16,
    pub y: i16,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
enum Direction {
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
    pub fn x(&self) -> f64 {
        match self {
            Direction::North | Direction::NorthEast | Direction::NorthWest => 1.,
            Direction::South | Direction::SouthEast | Direction::SouthWest => -1.,
            _ => 0.,
        }
    }

    pub fn y(&self) -> f64 {
        match self {
            Direction::East | Direction::NorthEast | Direction::SouthEast => 1.,
            Direction::West | Direction::NorthWest | Direction::SouthWest => -1.,
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
